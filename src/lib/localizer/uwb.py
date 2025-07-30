from typing import final
from src.lib.recorder import DataRecorderProtocol
from src.type import Position


class UWBLocalizer(DataRecorderProtocol):
    """
    UWB による位置推定のためのクラス
    """

    @final
    def estimate_uwb(self) -> Position | None:
        raise NotImplementedError("UWBLocalizer が実装されていません。")
