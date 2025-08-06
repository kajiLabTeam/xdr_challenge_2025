from typing import final
from src.lib.recorder import DataRecorderProtocol
from src.lib.recorder._orientation import QOrientationWithTimestamp
from src.lib.safelist._safelist import SafeList
from src.type import Position


class VIOLocalizer(DataRecorderProtocol):
    """
    VIO による位置推定のためのクラス
    """

    _viso_init_pos: Position | None = None
    _viso_orientations: SafeList[QOrientationWithTimestamp] = SafeList()

    @final
    def estimate_viso(self) -> Position | None:
        raise NotImplementedError("VISOLocalizer が実装されていません。")
