from typing import TypedDict
from src.localizer.base import LocalizerBase


class AcceData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    acc_x: float
    acc_y: float
    acc_z: float
    accuracy: float


class AcceLocalizer(LocalizerBase[AcceData]):
    key = "ACCE"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "acc_x": float,
        "acc_y": float,
        "acc_z": float,
        "accuracy": float,
    }
