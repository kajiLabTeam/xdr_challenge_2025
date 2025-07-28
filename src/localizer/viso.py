from typing import TypedDict
from localizer.base import LocalizerBase


class VisoData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    location_x: float
    location_y: float
    location_z: float
    quat_x: float
    quat_y: float
    quat_z: float


class VisoLocalizer(LocalizerBase[VisoData]):
    key = "VISO"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "location_x": float,
        "location_y": float,
        "location_z": float,
        "quat_x": float,
        "quat_y": float,
        "quat_z": float,
    }
    pass
