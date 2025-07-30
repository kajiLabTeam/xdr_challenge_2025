from typing import TypedDict
from ._base import BaseDataRecorder


class GyroData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    gyr_x: float
    gyr_y: float
    gyr_z: float
    accuracy: float


class GyroDataRecorder(BaseDataRecorder[GyroData]):
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
