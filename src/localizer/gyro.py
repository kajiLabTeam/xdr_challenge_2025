from typing import TypedDict
from localizer.base import LocalizerBase


class GyroData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    gyr_x: float
    gyr_y: float
    gyr_z: float
    accuracy: float


class GyroLocalizer(LocalizerBase[GyroData]):
    key = "GYRO"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "gyr_x": float,
        "gyr_y": float,
        "gyr_z": float,
        "accuracy": float,
    }
    pass
