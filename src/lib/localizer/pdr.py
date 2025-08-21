from typing import Protocol, final
from src.lib.recorder import DataRecorderProtocol
from src.type import Position


class PDRLocalizer(DataRecorderProtocol, Protocol):
    """
    PDR による位置推定のためのクラス
    """

    def estimate_pdr(self) -> Position: ...
