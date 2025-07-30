from typing import Protocol, final
from src.lib.safelist._safelist import SafeList
from src.type import Position


class PositionDataRecorderProtocol(Protocol):
    positions: SafeList[Position]

    def __getitem__(self, index: int) -> Position | None:
        raise NotImplementedError("This method should be implemented by subclasses.")


class PositionDataRecorder:
    positions: SafeList[Position] = SafeList([])

    @final
    def __getitem__(self, index: int) -> Position | None:
        try:
            return self.positions[index]
        except IndexError:
            return None
