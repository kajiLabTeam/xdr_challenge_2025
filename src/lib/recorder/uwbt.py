from typing import TypedDict, final
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
        "nlos": lambda x: x == "1.0",
    }

    _tag_ids: set[str] = set()

    @final
    @property
    def tag_ids(self) -> set[str]:
        if len(self._tag_ids) >= 3:
            return self._tag_ids

        for data in self.last_appended_data:
            self._tag_ids.add(data["tag_id"])

        return self._tag_ids
