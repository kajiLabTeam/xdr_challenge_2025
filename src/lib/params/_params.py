from typing import Literal
import numpy as np
from src.services.env import set_env
from src.lib.decorators.env_or_call import (
    env_exists,
    float_env_or_call,
    bool_env_or_call,
    str_env_or_call,
)


class Params:
    @staticmethod
    @bool_env_or_call("DEMO")
    def demo() -> bool:
        """
        デモモードか
        """
        raise Exception(
            "DEMO 環境変数が設定されていません。main.py で設定する必要があります。"
        )

    @staticmethod
    @str_env_or_call("ESTIMATE_MODE")
    def estimate_mode() -> Literal["COMPETITION", "ONLY_PDR", "ONLY_VIO", "ONLY_UWB"]:
        """
        推定時のモード
        """
        return "COMPETITION"

    @staticmethod
    @float_env_or_call("WINDOW_ACC_SEC")
    def window_acc_sec() -> float:
        """
        PDRにおける平滑化時の加速度センサのウィンドウ幅 (秒)
        """
        return 1.0

    @staticmethod
    @float_env_or_call("STEP")
    def step() -> float:
        """
        PDRにおけるステップ幅の固定値 (メートル)
        TODO: 動的に計算するようにする
        """
        return 0.4

    @staticmethod
    @float_env_or_call("EXTREMA_DISTANCE_SEC")
    def extrema_distance_sec() -> float:
        """
        PDRにおける極大・極小の検出のための距離閾値 (秒)
        """
        return 0.5

    @staticmethod
    @float_env_or_call("PEAK_HEIGHT")
    def peak_height() -> float:
        """
        PDRにおけるピーク(山)検出のための高さ閾値(m/s^2)
        """
        return 1.0

    @staticmethod
    @float_env_or_call("TROUGH_HEIGHT")
    def trough_height() -> float:
        """
        PDRにおけるトラフ(谷)検出のための高さ閾値(m/s^2)
        """
        return -5.0

    @staticmethod
    @float_env_or_call("INIT_ANGLE_RAD")
    def init_angle_rad() -> float:
        """
        PDRにおける初期角度 (ラジアン)
        TODO: 動的に計算するようにする
        """
        return np.deg2rad(80)

    @staticmethod
    @float_env_or_call("STRIDE_SCALE")
    def stride_scale() -> float:
        """
        PDRにおける歩幅計算のためのスケール係数
        """
        return 0.9

    @staticmethod
    @float_env_or_call("STRIDE_THRESHOLD")
    def stride_threshold() -> float:
        """
        PDRにおける歩幅計算のための閾値
        """
        return 0.035

    @staticmethod
    @float_env_or_call("UWB_FAR_DISTANCE")
    def uwb_far_distance() -> float:
        """
        UWBにおける遠距離の閾値 (メートル)
        """
        return 3.0

    @staticmethod
    @float_env_or_call("UWB_NLOS_FACTOR")
    def uwb_nlos_factor() -> float:
        """
        UWBにおけるNLOSの影響を考慮するための係数
        """
        return 0.25

    @staticmethod
    @env_exists
    def set_param(env_name: str, value: str | float | bool) -> None:
        """
        環境変数を設定する
        """
        set_env(env_name, value)
