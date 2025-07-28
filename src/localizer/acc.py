from typing import TypedDict
from localizer.base import LocalizerBase


class AccData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    acc_x: float
    acc_y: float
    acc_z: float
    accuracy: float


class AccLocalizer(LocalizerBase[AccData]):,
    pass
