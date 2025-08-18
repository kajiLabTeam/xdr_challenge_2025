from src.lib.recorder.viso import VisoData
from src.type import Position, PositionWithTimestamp


class VisoService:
    @staticmethod
    def to_position(data: VisoData) -> Position:
        """
        VISO データから Position オブジェクトを生成する
        """
        return Position(
            x=data["location_x"],
            y=data["location_y"],
            z=data["location_z"],
        )

    @staticmethod
    def to_position_with_timestamp(data: VisoData) -> PositionWithTimestamp:
        """
        VISO データから PositionWithTimestamp オブジェクトを生成する
        """
        return PositionWithTimestamp(
            x=data["location_x"],
            y=data["location_y"],
            z=data["location_z"],
            timestamp=data["app_timestamp"],
        )
