from typing import final
from src.lib.decorators.attr_check import require_attr_appended
from src.lib.recorder import DataRecorder
from src.lib.visualizer._visualizer import Visualizer
from src.type import Position
from .pdr import PDRLocalizer
from .viso import VISOLocalizer
from .uwb import UWBLocalizer


class Localizer(DataRecorder, Visualizer, PDRLocalizer, VISOLocalizer, UWBLocalizer):
    """
    位置推定のためのクラス
    """

    @final
    @require_attr_appended("positions", 1)
    def estimate(self) -> None:
        # pdr_pos = self.estimate_pdr()
        # uwb_pos = self.estimate_uwb()
        # viso_pos = self.estimate_viso()

        last_pos = self.positions[-1] or Position(0.0, 0.0, 0.0)
        pos = Position(
            x=last_pos.x + 1,
            y=last_pos.y,
            z=last_pos.z,
        )

        # 推定結果を保存
        self.positions.append(pos)
