from typing import TypedDict
from src.localizer.base import LocalizerBase


class MagnData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    mag_x: float
    mag_y: float
    mag_z: float
    accuracy: float


class MagnLocalizer(LocalizerBase[MagnData]):
    key = "MAGN"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "mag_x": float,
        "mag_y": float,
        "mag_z": float,
        "accuracy": float,
    }
    pass
