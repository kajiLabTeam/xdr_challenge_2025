from typing import TypedDict
from src.lib.safelist._safelist import SafeList
from src.type import Position, QOrientationWithTimestamp
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
    quat_w: float


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
        "quat_w": float,
    }

    def last_applended_orientations(self) -> SafeList[QOrientationWithTimestamp]:
        """
        最後に追加されたクォータニオンデータを取得するメソッド
        Returns:
            list[QOrientationWithTimestamp]: クォータニオンデータのリスト
        """
        print(f"→→ {len(self.last_appended_data)}")
        if not self.last_appended_data:
            return SafeList()

        last_data = map(
            lambda data: QOrientationWithTimestamp(
                timestamp=data["sensor_timestamp"],
                x=data["quat_y"],
                y=data["quat_z"],
                z=data["app_timestamp"],
                w=data["quat_x"],
            ),
            self.last_appended_data,
        )

        return SafeList(*last_data)

    def last_applended_positions(self) -> SafeList[Position]:
        """
        最後に追加された位置データを取得するメソッド
        Returns:
            SafeList[Position]: 位置データのリスト
        """
        if not self.__last_appended_data:
            return SafeList()

        last_data = map(
            lambda data: Position(
                x=data["location_x"],
                y=data["location_y"],
                z=data["location_z"],
            ),
            self.last_appended_data,
        )

        return SafeList(*last_data)
