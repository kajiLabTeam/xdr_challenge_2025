from logging import Logger
from typing import Protocol, final
from src.lib.safelist._safelist import SafeList
from src.type import TimedPose


class PositionDataRecorderProtocol(Protocol):
    poses: SafeList[TimedPose]

    def __init__(self, trial_id: str, logger: Logger): ...
    @property
    def first_pose(self) -> TimedPose: ...
    @property
    def last_pose(self) -> TimedPose: ...
    def __getitem__(self, index: int) -> TimedPose | None: ...


class PositionDataRecorder:
    def __init__(self, trial_id: str, logger: Logger):
        self.poses: SafeList[TimedPose] = SafeList()

    @final
    @property
    def first_pose(self) -> TimedPose:
        """
        最初の位置を取得するメソッド
        """
        first_pos = self.poses[0]

        if first_pos is None:
            raise IndexError("初期位置がありません")

        return first_pos

    @final
    @property
    def last_pose(self) -> TimedPose:
        """
        最後の位置を取得するメソッド
        """
        last_pos = self.poses[-1]

        if last_pos is None:
            raise IndexError("位置データがありません")

        return last_pos

    @final
    def __getitem__(self, index: int) -> TimedPose | None:
        try:
            return self.poses[index]
        except IndexError:
            return None
