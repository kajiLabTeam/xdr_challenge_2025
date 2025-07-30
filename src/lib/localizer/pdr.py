from typing import final
from src.lib.recorder import DataRecorderProtocol
from src.type import Position


class PDRLocalizer(DataRecorderProtocol):
    """
    PDR による位置推定のためのクラス
    """

    @final
    def estimate_pdr(self) -> Position:
        raise NotImplementedError("PDRLocalizer が実装されていません。")
