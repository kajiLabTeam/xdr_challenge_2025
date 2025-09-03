from functools import cache
from logging import Logger
from typing import TypedDict, final
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

    def __init__(self, trial_id: str, logger: Logger) -> None:
        super().__init__(trial_id, logger)

        self._tag_ids: set[str] = set()

    @final
    @property
    def tag_ids(self) -> set[str]:
        if len(self._tag_ids) >= 3:
            return self._tag_ids

        for data in self.last_appended_data:
            self._tag_ids.add(data["tag_id"])

        return self._tag_ids
