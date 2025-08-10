from typing import final
from src.lib.decorators.attr_check import require_attr_appended
from src.lib.recorder import DataRecorder
from src.lib.visualizer._visualizer import Visualizer
from src.type import Position
from .pdr import PDRLocalizer
from .vio import VIOLocalizer
from .uwb import UWBLocalizer


class Localizer(DataRecorder, Visualizer, PDRLocalizer, VIOLocalizer, UWBLocalizer):
    """
    位置推定のためのクラス
    """

    @final
    @require_attr_appended("positions", 1)
    def estimate(self) -> None:
        # pdr_pos = self.estimate_pdr()
        uwb_pos = self.estimate_uwb()
        # viso_pos = self.estimate_vio()

        # UWBの推定結果が得られた場合はそれを使用、そうでなければ前回の位置を使用
        if uwb_pos is not None:
            self.positions.append(uwb_pos)
        else:
            last_pos = self.last_position()
            # 推定結果を保存（暫定的に+1）
            self.positions.append(Position(last_pos.x + 1, last_pos.y, last_pos.z))
