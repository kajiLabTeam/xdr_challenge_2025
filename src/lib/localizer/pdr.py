from typing import final
import numpy as np
from scipy import signal
from src.lib.params._params import Params
from src.lib.recorder import DataRecorderProtocol
from src.type import EstimateResult, Position, PositionWithTimestamp


class PDRLocalizer(DataRecorderProtocol):
    """
    PDR による位置推定のためのクラス
    """

    def __init__(self) -> None:
        self._pdr_start_timestamp: float = 0.0
        self._pdr_init_pos: Position | None = None
        self._pdr_init_direction: float | None = None

    @final
    def estimate_pdr(self) -> EstimateResult:
        """
        PDR による位置推定を行う
        """
        if self._pdr_init_pos is None or self._pdr_init_direction is None:
            raise ValueError("PDR の開始位置または開始方向が設定されていません。")

        trajectory = self._pdr_estimate_pdr_within_range(
            init_pos=self._pdr_init_pos,
            init_direction=self._pdr_init_direction,
            start_time=self._pdr_start_timestamp,
        )

        if len(trajectory) == 0:
            return (Position(0, 0, 0), 0.0)

        last_z = self.last_position.z
        position = Position(x=trajectory[-1].x, y=trajectory[-1].y, z=last_z)
        return (position, 1.0)

    @final
    def _pdr_estimate_pdr_within_range(
        self,
        init_pos: Position,
        init_direction: float,
        start_time: float,
    ) -> list[PositionWithTimestamp]:
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
        trajectory: list[PositionWithTimestamp] = [
            PositionWithTimestamp(timestamp=0, x=init_pos.x, y=init_pos.y, z=init_pos.z)
        ]

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
            trajectory.append(PositionWithTimestamp(timestamp=0, x=x, y=y, z=0))

        return trajectory

    @final
    def switch_to_pdr(
        self, timestamp: float, init_pos: Position, init_direction: float
    ) -> None:
        """
        PDR切り替え時の初期設定を行う
        """
        self._pdr_start_timestamp = timestamp
        self._pdr_init_pos = init_pos
        self._pdr_init_direction = init_direction
