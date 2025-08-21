from typing import TypedDict
from ._base import BaseDataRecorder


class GyroData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    gyr_x: float
    gyr_y: float
    gyr_z: float
    accuracy: float


class GyroDataRecorder(BaseDataRecorder[GyroData]):
    key = "GYRO"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "gyr_x": float,
        "gyr_y": float,
        "gyr_z": float,
        "accuracy": float,
    }

    @property
    def fs(self) -> float:
        if len(self.data) < 2:
            return 0.0

        return len(self.data) / (
            self.data[-1]["app_timestamp"] - self.data[0]["app_timestamp"]
        )
