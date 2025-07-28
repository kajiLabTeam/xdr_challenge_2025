from typing import TypedDict
from src.localizer.base import LocalizerBase


class UwbPData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    tag_id: str
    distance: float
    aoa_azimuth: float
    aoa_elevation: float
    nlos: bool


class UwbTLocalizer(LocalizerBase[UwbPData]):
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
    pass
