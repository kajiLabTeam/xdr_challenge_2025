from typing import final
import numpy as np
from src.lib.recorder import DataRecorderProtocol
from src.type import Position

window_acc_sec = 1.0  # 加速度の移動平均フィルタのウィンドウサイズ（秒）
window_gyro_sec = 1.0  # 角速度の移動平均フィルタのウィンドウサイズ（秒）
step = 0.4  # 歩幅（メートル）
peak_distance_sec = 0.5  # ピーク検出の最小距離（秒）
peak_height = 1.0  # ピーク検出の最小高さ
init_angle = np.deg2rad(80)  # 初期角度（ラジアン）


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
            acce_df["acce_x"] ** 2 + acce_df["acce_y"] ** 2 + acce_df["acce_z"] ** 2
        )

        # 角度の計算
        gyro_df["angle"] = np.cumsum(gyro_df["x"]) / gyro_fs

        # 移動平均フィルタ
        window_acc_frame = int(window_acc_sec * acce_fs)
        window_gyro_frame = int(window_gyro_sec * gyro_fs)
        acce_df["low_norm"] = acce_df["norm"].rolling(window=window_acc_frame).mean()
        gyro_df["low_angle"] = gyro_df["angle"].rolling(window=window_gyro_frame).mean()

        return Position(0, 0, 0)
