from typing import final
import numpy as np
from scipy import signal
from src.lib.params._params import Params
from src.lib.recorder import DataRecorderProtocol
from src.type import Position


class PDRLocalizer(DataRecorderProtocol):
    """
    PDR による位置推定のためのクラス
    """

    @final
    def estimate_pdr(self) -> Position:
        """
        PDR による位置推定を行う
        """

        # サンプリング周波数の計算
        acce_fs = self.acc_datarecorder.fs
        gyro_fs = self.gyro_datarecorder.fs

        # データフレームの取得
        acce_df = self.acc_datarecorder.df
        gyro_df = self.gyro_datarecorder.df

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
        first_position = self.positions[0]
        if first_position is None:
            raise ValueError("初期位置が設定されていません")

        track: list[Position] = [first_position]

        for peak in peaks:
            time = acce_df["app_timestamp"][peak]
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

            x = (
                Params.step()
                * np.cos(gyro_df["angle"][gyro_i] + Params.init_angle_rad())
                + track[-1][0]
            )
            y = (
                Params.step()
                * np.sin(gyro_df["angle"][gyro_i] + Params.init_angle_rad())
                + track[-1][1]
            )

            track.append(Position(x, y, 0))

        return Position(0, 0, 0)
