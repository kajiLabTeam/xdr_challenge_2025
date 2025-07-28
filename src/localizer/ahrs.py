from typing import TypedDict
from localizer.base import LocalizerBase


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


class AhrsLocalizer(LocalizerBase[AhrsData]):
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

    pass
