from typing import final
from src.lib.decorators.attr_check import require_attr_appended
from src.lib.recorder import DataRecorder
from src.lib.visualizer import Visualizer
from src.type import Position
from .pdr import PDRLocalizer
from .vio import VIOLocalizer
from .uwb import UWBLocalizer


class Localizer(
    DataRecorder,
    Visualizer,
    PDRLocalizer,
    VIOLocalizer,
    UWBLocalizer,
):
    """
    位置推定のためのクラス
    """

    @final
    @require_attr_appended("positions", 1)
    def estimate(self) -> None:
        # pdr_pos = self.estimate_pdr()
        # uwb_pos = self.estimate_uwb()
        vis_pos = self.estimate_vio()
        # viso_orientations = self.estimate_vio_orientations()

        last_pos = self.last_position()

        # 推定結果を保存
        self.positions.append(vis_pos if vis_pos else Position(0, 0, 0))
