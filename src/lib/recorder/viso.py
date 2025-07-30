from typing import TypedDict
from ._base import BaseDataRecorder


class VisoData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    location_x: float
    location_y: float
    location_z: float
    quat_x: float
    quat_y: float
    quat_z: float


class VisoDataRecorder(BaseDataRecorder[VisoData]):
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
