from typing import final
from src.lib.decorators.demo_only import demo_only
from src.lib.decorators.attr_check import require_attr_appended
from src.lib.params._params import Params
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
        """
        位置推定を行う
        """
        if Params.estimate_mode() == "COMPETITION":
            self._estimate_for_competition()
            return

        if Params.estimate_mode() == "ONLY_PDR":
            self._estimate_only_pdr()
            return

        if Params.estimate_mode() == "ONLY_VIO":
            self._estimate_only_vio()
            return

        if Params.estimate_mode() == "ONLY_UWB":
            self._estimate_only_uwb()
            return

    @final
    def _estimate_for_competition(self) -> None:
        """
        競技用の位置推定を行う。
        TODO: 実装
        """
        # pdr_pos = self.estimate_pdr()

        # 青色のみ推定を優先的に使用
        # uwb_pos = self.estimate_uwb()
        viso_pos = self.estimate_vio()

        # UWBの推定結果が得られた場合はそれを使用、そうでなければ前回の位置を使用
        last_pos = self.last_position()
        self.positions.append(viso_pos if viso_pos else last_pos)

    @final
    @demo_only
    def _estimate_only_pdr(self) -> None:
        """
        PDR のみを使用して位置を推定する(demo用)
        """
        pdr_pos = self.estimate_pdr()
        self.positions.append(pdr_pos)

    @final
    @demo_only
    def _estimate_only_vio(self) -> None:
        """
        VIO のみを使用して位置を推定する(demo用)
        """
        vio_pos = self.estimate_vio()
        self.positions.append(vio_pos if vio_pos else Position(0, 0, 0))

    @final
    @demo_only
    def _estimate_only_uwb(self) -> None:
        """
        UWB のみを使用して位置を推定する(demo用)
        """
        uwb_pos = self.estimate_uwb()
        self.positions.append(uwb_pos if uwb_pos else Position(0, 0, 0))
