from logging import Logger
from typing import cast
from ..type import ALLOWED_SENSOR_TYPES, SensorData, SensorType
from .acce import AcceLocalizer
from .ahrs import AhrsLocalizer
from .gpos import GposLocalizer
from .gyro import GyroLocalizer
from .magn import MagnLocalizer
from .uwbp import UwbPLocalizer
from .uwbt import UwbTLocalizer
from .viso import VisoLocalizer


class Localizer:
    """
    位置推定のためのクラス
    """

    def __init__(self, trial_id: str, logger: Logger):
        """
        Args:
            trial_id (str): トライアルID
            logger (Logger): ロガー
        """
        self.trial_id = trial_id
        self.logger = logger
        self.acc_localizer = AcceLocalizer(self.trial_id, self.logger)
        self.gyro_localizer = GyroLocalizer(self.trial_id, self.logger)
        self.magn_localizer = MagnLocalizer(self.trial_id, self.logger)
        self.ahrs_localizer = AhrsLocalizer(self.trial_id, self.logger)
        self.uwbp_localizer = UwbPLocalizer(self.trial_id, self.logger)
        self.uwbt_localizer = UwbTLocalizer(self.trial_id, self.logger)
        self.gpos_localizer = GposLocalizer(self.trial_id, self.logger)
        self.viso_localizer = VisoLocalizer(self.trial_id, self.logger)

    def set_sensor_data(self, sensor_data: SensorData) -> None:
        """
        センサーデータを保存するメソッド

        Args:
            sensor_data (SensorData): センサーデータ
        """
        for row in sensor_data:
            sensor_type = row[0]
            data = row[1]
            if sensor_type == "ACCE":
                self.acc_localizer.append(sensor_type, data)
            elif sensor_type == "GYRO":
                self.gyro_localizer.append(sensor_type, data)
            elif sensor_type == "MAGN":
                self.magn_localizer.append(sensor_type, data)
            elif sensor_type == "AHRS":
                self.ahrs_localizer.append(sensor_type, data)
            elif sensor_type == "UWBP":
                self.uwbp_localizer.append(sensor_type, data)
            elif sensor_type == "UWBT":
                self.uwbt_localizer.append(sensor_type, data)
            elif sensor_type == "GPOS":
                self.gpos_localizer.append(sensor_type, data)
            elif sensor_type == "VISO":
                self.viso_localizer.append(sensor_type, data)
            else:
                self.logger.error(f"Unknown sensor type: {sensor_type}")
