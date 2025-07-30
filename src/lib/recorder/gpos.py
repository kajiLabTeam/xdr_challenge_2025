from typing import TypedDict, final

import pandas as pd
from ._base import BaseDataRecorder


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


class GposDataRecorder(BaseDataRecorder[GposData]):
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
