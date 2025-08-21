from typing import Protocol, final
from src.lib.recorder import DataRecorderProtocol
from src.type import Position


class PDRLocalizer(DataRecorderProtocol, Protocol):
    """
    PDR による位置推定のためのクラス
    """

    def estimate_pdr(self) -> Position | None:
        """
        PDR による位置推定を行う
        Returns:
            Position | None: 推定された位置、または推定できなかった場合は None
        """
        return None
