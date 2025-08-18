from src.lib.recorder.gpos import GposData
from src.type import Position, PositionWithTimestamp


class GposService:
    @staticmethod
    def to_position(data: GposData) -> Position:
        """
        GPOS データから Position オブジェクトを生成する
        """
        return Position(
            x=data["location_x"],
            y=data["location_y"],
            z=data["location_z"],
        )

    @staticmethod
    def to_position_with_timestamp(data: GposData) -> PositionWithTimestamp:
        """
        GPOS データから PositionWithTimestamp オブジェクトを生成する
        """
        return PositionWithTimestamp(
            x=data["location_x"],
            y=data["location_y"],
            z=data["location_z"],
            timestamp=data["app_timestamp"],
        )
