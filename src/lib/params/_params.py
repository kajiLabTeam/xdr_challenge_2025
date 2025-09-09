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
    @float_env_or_call("WINDOW_GYRO_SEC")
    def window_gyro_sec() -> float:
        """
        PDRにおける平滑化時のジャイロセンサのウィンドウ幅 (秒)
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
    @float_env_or_call("PEAK_DISTANCE_SEC")
    def peak_distance_sec() -> float:
        """
        PDRにおけるピーク検出のための距離閾値 (秒)
        """
        return 0.5

    @staticmethod
    @float_env_or_call("PEAK_HEIGHT")
    def peak_height() -> float:
        """
        PDRにおけるピーク検出のための高さ閾値 (メートル)
        """
        return 1.0

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
        return 0.3

    @staticmethod
    @float_env_or_call("UWB_TIME_DIFF_K")
    def uwb_time_diff_k() -> float:
        """
        UWBにおける時刻差精度計算のシグモイド勾配パラメータ
        """
        return 7.0

    @staticmethod
    @float_env_or_call("UWB_TIME_DIFF_X0")
    def uwb_time_diff_x0() -> float:
        """
        UWBにおける時刻差精度計算のシグモイドシフトパラメータ
        """
        return 0.2

    @staticmethod
    @float_env_or_call("UWB_DISTANCE_K")
    def uwb_distance_k() -> float:
        """
        UWBにおける距離精度計算のシグモイド勾配パラメータ
        """
        return 3.0

    @staticmethod
    @float_env_or_call("UWB_DISTANCE_X0")
    def uwb_distance_x0() -> float:
        """
        UWBにおける距離精度計算のシグモイドシフトパラメータ
        """
        return 1.5

    @staticmethod
    @float_env_or_call("YAW_ADJUST")
    def yaw_adjust() -> float:
        """
        GPOSのyaw角に対する調整値 (ラジアン)
        """
        return np.deg2rad(-5)

    @staticmethod
    @env_exists
    def set_param(env_name: str, value: str | float | bool) -> None:
        """
        環境変数を設定する
        """
        set_env(env_name, value)
