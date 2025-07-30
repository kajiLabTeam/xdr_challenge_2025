from typing import final
from src.lib.recorder import DataRecorderProtocol
from src.type import Position


class VISOLocalizer(DataRecorderProtocol):
    """
    VISO による位置推定のためのクラス
    """

    @final
    def estimate_viso(self) -> Position | None:
        raise NotImplementedError("VISOLocalizer が実装されていません。")
