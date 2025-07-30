from typing import TypedDict
from ._base import BaseDataRecorder


class MagnData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    mag_x: float
    mag_y: float
    mag_z: float
    accuracy: float


class MagnDataRecorder(BaseDataRecorder[MagnData]):
    key = "MAGN"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "mag_x": float,
        "mag_y": float,
        "mag_z": float,
        "accuracy": float,
    }
