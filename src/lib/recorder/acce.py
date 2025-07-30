from typing import TypedDict
from src.lib.recorder._base import BaseDataRecorder


class AcceData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    acc_x: float
    acc_y: float
    acc_z: float
    accuracy: float


class AcceDataRecorder(BaseDataRecorder[AcceData]):
    key = "ACCE"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "acc_x": float,
        "acc_y": float,
        "acc_z": float,
        "accuracy": float,
    }
