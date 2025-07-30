from typing import TypedDict
from ._base import BaseDataRecorder


class AhrsData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    pitch_x: float
    roll_y: float
    yaw_z: float
    quat_2: float
    quat_3: float
    quat_4: float
    accuracy: float


class AhrsDataRecorder(BaseDataRecorder[AhrsData]):
    key = "AHRS"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "pitch_x": float,
        "roll_y": float,
        "yaw_z": float,
        "quat_2": float,
        "quat_3": float,
        "quat_4": float,
        "accuracy": float,
    }
