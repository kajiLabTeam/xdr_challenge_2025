from typing import TypedDict
from ._base import BaseDataRecorder


class UwbPData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    tag_id: str
    distance: float
    direction_vec_x: float
    direction_vec_y: float
    direction_vec_z: float


class UwbPDataRecorder(BaseDataRecorder[UwbPData]):
    key = "UWBP"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "tag_id": str,
        "distance": float,
        "direction_vec_x": float,
        "direction_vec_y": float,
        "direction_vec_z": float,
    }
    pass
