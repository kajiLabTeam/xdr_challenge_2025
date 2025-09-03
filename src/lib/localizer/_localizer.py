from typing import final
from src.lib.decorators.demo_only import demo_only
from src.lib.decorators.attr_check import require_attr_appended
from src.lib.decorators.time import timer
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
    @timer
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

        raise ValueError(f"ESTIMATE_MODE が正しくありません: {Params.estimate_mode()}")

    @final
    def _estimate_for_competition(self) -> None:
        """
        競技用の位置推定を行う
        """
        (pdr_pos, pdr_accuracy) = self.estimate_pdr()
        (uwb_pos, uwb_accuracy) = self.estimate_uwb()
        (vio_pos, vio_accuracy) = self.estimate_vio()

        # UWB の信頼度が 0.5 以上の場合は UWB を使用
        if uwb_accuracy >= 0.5 and uwb_pos:
            self.positions.append(uwb_pos)
            return

        # VIO の信頼度が 0.8 以上の場合は VIO を使用 TODO: 調整
        if vio_accuracy > 0.8:
            self.positions.append(vio_pos)
            return

        # PDR の信頼度が 0.8 以上の場合は PDR を使用 TODO: 調整
        if pdr_accuracy > 0.8:
            self.positions.append(pdr_pos)
            return

        last_pos = self.last_position()
        self.positions.append(vio_pos if vio_pos else last_pos)

    @final
    @demo_only
    def _estimate_only_pdr(self) -> None:
        """
        PDR のみを使用して位置を推定する(demo用)
        """
        (pdr_pos, pdr_accuracy) = self.estimate_pdr()
        self.positions.append(pdr_pos)

    @final
    @demo_only
    def _estimate_only_vio(self) -> None:
        """
        VIO のみを使用して位置を推定する(demo用)
        """
        (vio_pos, vio_accuracy) = self.estimate_vio()
        self.positions.append(vio_pos if vio_pos else self.last_position())

    @final
    @demo_only
    def _estimate_only_uwb(self) -> None:
        """
        UWB のみを使用して位置を推定する(demo用)
        total_confidenceが閾値を下回る場合はプロットしない
        """
        (uwb_pos, uwb_accuracy) = self.estimate_uwb()

        # total_confidenceの閾値を下回る場合は前の位置を維持
        if uwb_accuracy >= 0.3:
            self.positions.append(uwb_pos)
        else:
            # 信頼性が低い場合は前の位置を維持（プロットされない）
            self.positions.append(Position(0, 0, 0))
