from logging import Logger
from typing import cast
from type import ALLOWED_SENSOR_TYPES, SensorType
from .acc import AccLocalizer
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
        self.acc_localizer = AccLocalizer(self.trial_id, self.logger)
        self.gyro_localizer = GyroLocalizer(self.trial_id, self.logger)
        self.magn_localizer = MagnLocalizer(self.trial_id, self.logger)
        self.ahrs_localizer = AhrsLocalizer(self.trial_id, self.logger)
        self.uwbp_localizer = UwbPLocalizer(self.trial_id, self.logger)
        self.uwbt_localizer = UwbTLocalizer(self.trial_id, self.logger)
        self.gpos_localizer = GposLocalizer(self.trial_id, self.logger)
        self.viso_localizer = VisoLocalizer(self.trial_id, self.logger)

    def save(self, recv_data: str) -> None:
        """
        センサーデータを保存するメソッド

        Args:
            sensor_type (SensorType): センサーの種類
            data (list): センサーデータのリスト
        """
        rows = self._process_data(recv_data)
        for row in rows:
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

    def _process_data(self, recv_data: str) -> list[tuple[SensorType, list[str]]]:
        recv_sensor_lines = recv_data.splitlines()

        rows: list[tuple[SensorType, list[str]]] = []
        for line in recv_sensor_lines:
            if not line.strip():
                continue

            parts = line.strip().split(";")
            sensor_type = parts[0]
            data_row: list[str] = parts[1:]

            if sensor_type in ALLOWED_SENSOR_TYPES:
                rows.append((cast(SensorType, sensor_type), data_row))
            else:
                self.logger.error(f"{sensor_type}: 存在しないセンサー種類です。")

        return rows
