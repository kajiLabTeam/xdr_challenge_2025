from typing import final
import numpy as np
from scipy import signal
from src.lib.params._params import Params
from src.lib.recorder import DataRecorderProtocol
from src.type import EstimateResult, TimedPose


class PDRLocalizer(DataRecorderProtocol):
    """
    PDR による位置推定のためのクラス
    """

    @final
    def estimate_pdr(self) -> EstimateResult:
        """
        PDR による位置推定を行う
        """
        if self._pdr_init_pose is None or self._pdr_init_direction is None:
            raise ValueError("PDR の開始位置または開始方向が設定されていません。")

        trajectory = self._pdr_estimate_pdr_within_range(
            init_pose=self._pdr_init_pose,
            init_direction=self._pdr_init_direction,
            start_time=self._pdr_start_timestamp,
        )

        if len(trajectory) == 0:
            return (TimedPose(x=0, y=0, z=0, yaw=0, timestamp=self.timestamp), 0.0)

        return (trajectory[-1], 1.0)

    @final
    def _pdr_estimate_pdr_within_range(
        self,
        init_pose: TimedPose,
        init_direction: float,
        start_time: float,
    ) -> list[TimedPose]:
        """
        指定した時間範囲内での PDR による位置推定を行う
        """
        # サンプリング周波数の計算
        acce_fs = self.acc_datarecorder.fs
        gyro_fs = self.gyro_datarecorder.fs

        # start_timestamp 以降のデータフレームの取得
        acce_df = self.acc_datarecorder.df
        acce_df = acce_df[acce_df["app_timestamp"] >= start_time]
        gyro_df = self.gyro_datarecorder.df
        gyro_df = gyro_df[gyro_df["app_timestamp"] >= start_time]

        # ノルムの計算
        acce_df["norm"] = np.sqrt(
            acce_df["acc_x"] ** 2 + acce_df["acc_y"] ** 2 + acce_df["acc_z"] ** 2
        )

        # 角度の計算
        gyro_df["angle"] = np.cumsum(gyro_df["gyr_x"]) / gyro_fs
        # 端末の向きが反転している場合は角度を反転
        if self._pdr_inverted:
            gyro_df["angle"] = -gyro_df["angle"]
            self.logger.debug("PDR: 端末方向が反転設定のため角度を反転しました")

        # 移動平均フィルタ
        window_acc_frame = int(Params.window_acc_sec() * acce_fs)
        window_gyro_frame = int(Params.window_gyro_sec() * gyro_fs)
        acce_df["low_norm"] = acce_df["norm"].rolling(window=window_acc_frame).mean()
        gyro_df["low_angle"] = gyro_df["angle"].rolling(window=window_gyro_frame).mean()

        # ピーク検出
        distance_frame = int(Params.peak_distance_sec() * acce_fs)
        peaks, _ = signal.find_peaks(
            acce_df["low_norm"],
            distance=distance_frame,
            height=Params.peak_height(),
        )

        gyro_timestamps = np.asarray(gyro_df["app_timestamp"].values)
        trajectory: list[TimedPose] = [init_pose]

        # 加速度データから歩幅の推定
        detected_steps: list[float] = []
        acc_norm_values = acce_df["low_norm"].values
        for i in range(len(peaks)):
            start_idx = peaks[i - 1] if i > 0 else 0
            end_idx = peaks[i]

            range_acc = np.array(acc_norm_values[start_idx:end_idx])
            valid_range_acc = range_acc[~np.isnan(range_acc)]
            if len(valid_range_acc) == 0:
                stride = detected_steps[-1] if detected_steps else Params.stride_scale()
            else:
                max_acc = np.max(np.array(valid_range_acc))
                min_acc = np.min(np.array(valid_range_acc))
                if max_acc - min_acc < Params.stride_threshold():
                    stride = 0
                else:
                    stride = Params.stride_scale() * np.power(max_acc - min_acc, 0.25)
            detected_steps.append(stride)

        for i, peak in enumerate(peaks):
            time = acce_df["app_timestamp"].iloc[peak]
            idx = np.searchsorted(gyro_timestamps, time)
            if idx == 0:
                gyro_i = gyro_df.index[0]
            elif idx == len(gyro_timestamps):
                gyro_i = gyro_df.index[-1]
            else:
                before = gyro_timestamps[idx - 1]
                after = gyro_timestamps[idx]
                if abs(time - before) <= abs(time - after):
                    gyro_i = gyro_df.index[idx - 1]
                else:
                    gyro_i = gyro_df.index[idx]
            step: float = (
                detected_steps[i] if i < len(detected_steps) else detected_steps[-1]
            )
            x = (
                step * np.cos(gyro_df["angle"][gyro_i] + init_direction)
                + trajectory[-1].x
            )
            y = (
                step * np.sin(gyro_df["angle"][gyro_i] + init_direction)
                + trajectory[-1].y
            )
            trajectory.append(TimedPose(x=x, y=y, z=0, yaw=0, timestamp=self.timestamp))

        return trajectory

    @final
    def _detect_device_orientation(self) -> bool:
        """
        ジャイロスコープの累積回転から端末の向きを動的に推定する

        Returns:
            bool: True if device is inverted, False if normal orientation
        """
        # ジャイロデータが少ない場合はデフォルト値を返す
        if len(self.gyro_datarecorder.data) < 100:
            self.logger.warning(
                "ジャイロデータが不足しています。デフォルト設定を使用します"
            )
            return True  # 001.txtベースのデフォルト

        # 分析用データを取得（約20秒分のデータを使用）
        gyro_fs = self.gyro_datarecorder.fs  # サンプリング周波数
        target_duration = 20.0  # 20秒
        analysis_samples = min(
            int(gyro_fs * target_duration), len(self.gyro_datarecorder.data)
        )

        # 最小限のデータが必要
        if analysis_samples < 50:
            analysis_samples = min(50, len(self.gyro_datarecorder.data))

        gyro_data = self.gyro_datarecorder.data[:analysis_samples]

        # ジャイロのX軸（ヨー角速度）データを抽出
        gyr_x_values = [data["gyr_x"] for data in gyro_data]
        timestamps = [data["app_timestamp"] for data in gyro_data]

        if len(gyr_x_values) < 2:
            return True

        # 平均的な時間間隔を計算
        dt_mean = np.mean(np.diff(timestamps))

        # 累積回転角を計算（PDRで実際に使用される方法と同じ）
        cumulative_rotation = np.sum(gyr_x_values) * dt_mean
        cumulative_rotation_deg = np.degrees(cumulative_rotation)

        # セクション別分析（より詳細な判定のため）
        n_sections = 4
        section_size = len(gyr_x_values) // n_sections
        section_rotations = []

        for i in range(n_sections):
            start_idx = i * section_size
            end_idx = (
                (i + 1) * section_size if i < n_sections - 1 else len(gyr_x_values)
            )

            if end_idx > start_idx:
                section_gyr_x = gyr_x_values[start_idx:end_idx]
                section_rotation = np.sum(section_gyr_x) * dt_mean
                section_rotation_deg = np.degrees(section_rotation)
                section_rotations.append(section_rotation_deg)

        # 方向の一貫性を評価
        positive_sections = sum(1 for r in section_rotations if r > 10)
        negative_sections = sum(1 for r in section_rotations if r < -10)

        # 判定ロジック
        confidence_threshold = abs(cumulative_rotation_deg)

        if confidence_threshold < 30:
            # 回転が少ない場合：デフォルト設定を使用
            self.logger.info(
                f"回転量が少ないため({cumulative_rotation_deg:.1f}°)、デフォルト設定を使用"
            )
            return True  # 001.txtベースのデフォルト

        # 001.txtの基準パターン：負方向回転で _pdr_inverted = True
        # 負方向回転が優勢 → True、正方向回転が優勢 → False
        if negative_sections >= 3 or cumulative_rotation_deg < -50:
            inversion = True
            confidence = "HIGH" if negative_sections >= 3 else "MEDIUM"
        elif positive_sections >= 3 or cumulative_rotation_deg > 50:
            inversion = False
            confidence = "HIGH" if positive_sections >= 3 else "MEDIUM"
        else:
            # 混在している場合は累積値で判定
            inversion = cumulative_rotation_deg < 0
            confidence = "LOW"

        self.logger.info(
            f"端末向き自動判定: _pdr_inverted = {inversion} "
            f"(累積回転: {cumulative_rotation_deg:.1f}°, 信頼度: {confidence})"
        )

        return inversion

    @final
    def switch_to_pdr(
        self,
        timestamp: float,
        init_pose: TimedPose,
        init_direction: float | None = None,
    ) -> None:
        """
        PDR切り替え時の初期設定を行う
        """
        self._pdr_start_timestamp = timestamp
        self._pdr_init_pose = init_pose
        self._pdr_init_direction = init_direction if init_direction else init_pose.yaw

        # 端末の向き（ジャイロスコープの回転方向）を動的に推定
        self._pdr_inverted = self._detect_device_orientation()

        # コンペモード用：端末方向設定のログ出力
        orientation_status = "INVERTED" if self._pdr_inverted else "NORMAL"
        self.logger.info(
            f"PDRモード開始 - 端末方向: {orientation_status} "
            f"(_pdr_inverted={self._pdr_inverted})"
        )
