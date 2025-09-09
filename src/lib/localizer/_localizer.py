from typing import Literal, final

import numpy as np
from src.lib.decorators.demo_only import demo_only
from src.lib.decorators.attr_check import require_attr_appended
from src.lib.decorators.time import timer
from src.lib.params._params import Params
from src.lib.recorder import DataRecorder
from src.lib.visualizer import Visualizer
from src.type import TimedPose
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

    current_method: Literal["INIT", "PDR", "VIO", "UWB"] = "INIT"

    @final
    @require_attr_appended("poses", 1)
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
        (uwb_pose, uwb_accuracy) = self.estimate_uwb()

        if uwb_accuracy >= 0.5 and uwb_pose:
            self.current_method = "UWB"
            self.poses.append(uwb_pose)
            return

        if self.current_method != "VIO":
            self.switch_to_vio(self.last_pose)
        (vio_pose, vio_accuracy) = self.estimate_vio()
        if vio_accuracy > 0.8:
            self.current_method = "VIO"
            self.poses.append(vio_pose)
            return

        if self.current_method != "PDR":
            self.switch_to_pdr(
                self.timestamp, self.last_pose, np.deg2rad(270)
            )  # TODO:  初期進行方向の調整(第3引数を省略する最後の yaw を使う(多分))
            self.current_method = "PDR"
        (pdr_pose, pdr_accuracy) = self.estimate_pdr()

        if pdr_accuracy > 0.8:
            self.poses.append(pdr_pose)
            return

        self.poses.append(vio_pose if vio_pose else self.last_pose)

    @final
    @demo_only
    def _estimate_only_pdr(self) -> None:
        """
        PDR のみを使用して位置を推定する(demo用)
        """
        if self.current_method != "PDR":
            self.current_method = "PDR"
            self.switch_to_pdr(0, self.first_pose, Params.init_angle_rad())

        (pdr_pose, pdr_accuracy) = self.estimate_pdr()
        self.poses.append(pdr_pose)

    @final
    @demo_only
    def _estimate_only_vio(self) -> None:
        """
        VIO のみを使用して位置を推定する(demo用)
        """
        if self.current_method != "VIO":
            self.switch_to_vio(self.last_pose)
        (vio_pose, vio_accuracy) = self.estimate_vio()
        self.poses.append(vio_pose if vio_pose else self.last_pose)

    @final
    @demo_only
    def _estimate_only_uwb(self) -> None:
        """
        UWB のみを使用して位置を推定する(demo用)
        total_confidenceが閾値を下回る場合はプロットしない
        """
        (uwb_pose, uwb_accuracy) = self.estimate_uwb()

        if uwb_accuracy >= 0.3:
            self.poses.append(uwb_pose)
        else:
            self.poses.append(TimedPose(0, 0, 0, 0, self.timestamp))
