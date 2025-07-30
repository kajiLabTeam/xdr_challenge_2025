from typing import final
from src.lib.decorators.attr_check import require_attr_appended
from src.lib.recorder import DataRecorder
from .pdr import PDRLocalizer
from .viso import VISOLocalizer
from .uwb import UWBLocalizer


class Localizer(DataRecorder, PDRLocalizer, VISOLocalizer, UWBLocalizer):
    """
    位置推定のためのクラス
    """

    @final
    @require_attr_appended("positions", 1)
    def estimate(self) -> None:

        pdr_pos = self.estimate_pdr()
        uwb_pos = self.estimate_uwb()
        viso_pos = self.estimate_viso()

        # 推定結果を保存
        self.positions.append(pdr_pos)
