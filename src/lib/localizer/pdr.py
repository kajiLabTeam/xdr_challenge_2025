from typing import final
import numpy as np
import pandas as pd
from scipy import signal
from scipy.linalg import orthogonal_procrustes
from src.lib.params._params import Params
from src.lib.recorder import DataRecorderProtocol
from src.type import EstimateResult, Position, PositionWithTimestamp


class PDRLocalizer(DataRecorderProtocol):
    """
    PDR による位置推定のためのクラス
    """

    def __init__(self) -> None:
        self._pdr_start_timestamp: float = 0.0
        self._pdr_init_direction: float | None = None
        self._pdr_trajectory: list[PositionWithTimestamp] = []
        self._pdr_init_pos: Position | None = None

    @final
    def estimate_pdr(self) -> EstimateResult:
        """
        PDR による位置推定を行う
        """
        # init_pos = (
        #     self._pdr_init_pos if self._pdr_init_pos is not None else self.positions[0]
        # )
        if self._pdr_init_pos is not None:
            init_pos = self._pdr_init_pos
        else:
            first_pos = self.positions[0]

            if first_pos is None:
                raise ValueError("初期位置が設定されていません")

            init_pos = first_pos

        if self._pdr_init_direction is not None:
            init_direction = self._pdr_init_direction
        else:
            init_direction = self._pdr_calc_init_direction()

        trajectory = self._pdr_estimate_pdr_within_range(
            init_pos=init_pos,
            init_direction=init_direction,
            start_time=self._pdr_start_timestamp,
        )

        if len(trajectory) == 0:
            return (Position(0, 0, 0), 0.0)

        position = Position(x=trajectory[-1].x, y=trajectory[-1].y, z=trajectory[-1].z)
        return (position, 1.0)

    @final
    def _pdr_estimate_pdr_within_range(
        self,
        init_pos: Position,
        init_direction: float,
        start_time: float,
        end_time: float | None = None,
    ) -> list[PositionWithTimestamp]:
        """
        指定した時間範囲内での PDR による位置推定を行う
        """
        # サンプリング周波数の計算
        acce_fs = self.acc_datarecorder.fs
        gyro_fs = self.gyro_datarecorder.fs

        # self.start_timestamp以降のデータフレームの取得

        acce_df_ = self.acc_datarecorder.df
        gyro_df_ = self.gyro_datarecorder.df
        end_time = end_time or max(
            acce_df_["app_timestamp"].max(), gyro_df_["app_timestamp"].max()
        )

        acce_df = acce_df_[
            (acce_df_["app_timestamp"] >= start_time)
            & (acce_df_["app_timestamp"] <= end_time)
        ]
        gyro_df = gyro_df_[
            (gyro_df_["app_timestamp"] >= start_time)
            & (gyro_df_["app_timestamp"] <= end_time)
        ]

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
            PositionWithTimestamp(
                x=init_pos.x, y=init_pos.y, z=init_pos.z, timestamp=start_time
            )
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
                + trajectory[-1][0]
            )
            y = (
                step * np.sin(gyro_df["angle"][gyro_i] + init_direction)
                + trajectory[-1][1]
            )
            trajectory.append(PositionWithTimestamp(x=x, y=y, z=0, timestamp=time))

        return trajectory

    @final
    def switch_to_pdr(
        self, timestamp: float, init_pos: Position, init_direction: float | None = None
    ) -> None:
        """
        PDR切り替え時の初期設定を行う
        """
        self._pdr_init_pos = init_pos
        self._pdr_start_timestamp = timestamp

        if init_direction is None:
            self._pdr_init_direction = self._pdr_calc_init_direction()
        else:
            self._pdr_init_direction = init_direction

    @final
    def _pdr_calc_init_direction(self) -> float:
        """
        プロクラステス分析で初期方向を設定する
        Args:
            threshold (float): 初期方向を設定するための移動距離の閾値
        Returns:
            float: PDR の初期方向(ラジアン)
        """
        gpos_df = self.gpos_datarecorder.df
        trajectory = self._pdr_estimate_pdr_within_range(
            start_time=0,
            
        )

        angle_deg = np.degrees(angle_rad)

        # 初期方向を設定
        self._viso_init_direction = angle_rad
        self.logger.info(
            f"VISO の初期方向を設定: {angle_rad:.1f}rad ({angle_deg:.1f}deg)"
        )

        return angle_rad
