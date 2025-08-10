from typing import TypedDict
from ._base import BaseDataRecorder


class UwbTData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    tag_id: str
    distance: float
    aoa_azimuth: float
    aoa_elevation: float
    nlos: bool


class UwbTDataRecorder(BaseDataRecorder[UwbTData]):
    key = "UWBT"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "tag_id": str,
        "distance": float,
        "aoa_azimuth": float,
        "aoa_elevation": float,
        "nlos": bool,
    }
