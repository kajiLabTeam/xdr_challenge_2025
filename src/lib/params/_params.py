import numpy as np
from src.lib.decorators.env_or_call import float_env_or_call


class Params:
    @staticmethod
    @float_env_or_call("WINDOW_ACC_SEC")
    def window_acc_sec() -> float:
        return 1.0

    @staticmethod
    @float_env_or_call("WINDOW_GYRO_SEC")
    def window_gyro_sec() -> float:
        return 1.0

    @staticmethod
    @float_env_or_call("STEP")
    def step() -> float:
        return 0.4

    @staticmethod
    @float_env_or_call("PEAK_DISTANCE_SEC")
    def peak_distance_sec() -> float:
        return 0.5

    @staticmethod
    @float_env_or_call("PEAK_HEIGHT")
    def peak_height() -> float:
        return 1.0

    @staticmethod
    @float_env_or_call("INIT_ANGLE_RAD")
    def init_angle_rad() -> float:
        return np.deg2rad(80)
