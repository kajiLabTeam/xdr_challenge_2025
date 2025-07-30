from typing import final
from src.type import Position


class PositionDataRecorder:
    __positions: list[Position] = []

    @final
    def append_position(self, position: Position) -> None:
        self.__positions.append(position)

    @final
    def __getitem__(self, index: int) -> Position | None:
        try:
            return self.__positions[index]
        except IndexError:
            return None
