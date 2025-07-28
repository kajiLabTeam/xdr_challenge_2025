from typing import TypedDict
from localizer.base import LocalizerBase


class GposData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    object_id: str
    location_x: float
    location_y: float
    location_z: float
    quat_x: float
    quat_y: float
    quat_z: float


class GposLocalizer(LocalizerBase[GposData]):
    key = "GPOS"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "object_id": str,
        "location_x": float,
        "location_y": float,
        "location_z": float,
        "quat_x": float,
        "quat_y": float,
        "quat_z": float,
    }
    pass
