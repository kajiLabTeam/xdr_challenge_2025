from typing import final
from src.type import Position


class PositionDataRecorder:
    positions: list[Position] = []

    @final
    def __getitem__(self, index: int) -> Position | None:
        try:
            return self.positions[index]
        except IndexError:
            return None
