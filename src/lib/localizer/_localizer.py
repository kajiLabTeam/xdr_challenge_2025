from typing import final
from src.lib.decorators.attr_check import require_attr_appended
from src.lib.recorder import DataRecorder


class Localizer(DataRecorder):
    """
    位置推定のためのクラス
    """

    @final
    @require_attr_appended("positions")
    def estimate(self) -> None:

        # self.positions.append()
        pass
