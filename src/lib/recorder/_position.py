from logging import Logger
from typing import Protocol, final
from src.lib.safelist._safelist import SafeList
from src.type import Position


class PositionDataRecorderProtocol(Protocol):
    positions: SafeList[Position]

    def __init__(self, trial_id: str, logger: Logger): ...
    @property
    def first_position(self) -> Position: ...
    @property
    def last_position(self) -> Position: ...
    def __getitem__(self, index: int) -> Position | None: ...


class PositionDataRecorder:
    def __init__(self, trial_id: str, logger: Logger):
        self.positions: SafeList[Position] = SafeList()

    @final
    @property
    def first_position(self) -> Position:
        """
        最初の位置を取得するメソッド
        """
        first_pos = self.positions[0]

        if first_pos is None:
            raise IndexError("初期位置がありません")

        return first_pos

    @final
    @property
    def last_position(self) -> Position:
        """
        最後の位置を取得するメソッド
        """
        last_pos = self.positions[-1]

        if last_pos is None:
            raise IndexError("位置データがありません")

        return last_pos

    @final
    def __getitem__(self, index: int) -> Position | None:
        try:
            return self.positions[index]
        except IndexError:
            return None
