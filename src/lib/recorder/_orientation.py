from logging import Logger
from typing import Protocol, final
from src.lib.safelist._safelist import SafeList
from src.type import QOrientationWithTimestamp


class OrientationDataRecorderProtocol(Protocol):
    orientations: SafeList[QOrientationWithTimestamp]

    def __init__(self, trial_id: str, logger: Logger): ...
    def last_orientation(self) -> QOrientationWithTimestamp: ...


class OrientationDataRecorder:
    def __init__(self, trial_id: str, logger: Logger):
        self.orientations: SafeList[QOrientationWithTimestamp] = SafeList()

    @final
    def last_orientation(self) -> QOrientationWithTimestamp:
        """
        最後の姿勢を取得するメソッド
        """
        last_orientation = self.orientations[-1]

        if last_orientation is None:
            raise IndexError("姿勢データがありません")

        return last_orientation
