from logging import Logger
from src.type import SensorData
from ._position import PositionDataRecorder
from .acce import AcceDataRecorder
from .ahrs import AhrsDataRecorder
from .gpos import GposDataRecorder
from .gyro import GyroDataRecorder
from .magn import MagnDataRecorder
from .uwbp import UwbPDataRecorder
from .uwbt import UwbTDataRecorder
from .viso import VisoDataRecorder


class DataRecorder(PositionDataRecorder):
    """
    データの記録を行うクラス
    """

    def __init__(self, trial_id: str, logger: Logger):
        """
        Args:
            trial_id (str): トライアルID
            logger (Logger): ロガー
        """
        super().__init__()

        self.trial_id = trial_id
        self.logger = logger
        self.acc_datarecorder = AcceDataRecorder(trial_id, logger)
        self.gyro_datarecorder = GyroDataRecorder(trial_id, logger)
        self.magn_datarecorder = MagnDataRecorder(trial_id, logger)
        self.ahrs_datarecorder = AhrsDataRecorder(trial_id, logger)
        self.uwbp_datarecorder = UwbPDataRecorder(trial_id, logger)
        self.uwbt_datarecorder = UwbTDataRecorder(trial_id, logger)
        self.gpos_datarecorder = GposDataRecorder(trial_id, logger)
        self.viso_datarecorder = VisoDataRecorder(trial_id, logger)

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
                self.acc_datarecorder.append(sensor_type, data)
            elif sensor_type == "GYRO":
                self.gyro_datarecorder.append(sensor_type, data)
            elif sensor_type == "MAGN":
                self.magn_datarecorder.append(sensor_type, data)
            elif sensor_type == "AHRS":
                self.ahrs_datarecorder.append(sensor_type, data)
            elif sensor_type == "UWBP":
                self.uwbp_datarecorder.append(sensor_type, data)
            elif sensor_type == "UWBT":
                self.uwbt_datarecorder.append(sensor_type, data)
            elif sensor_type == "GPOS":
                self.gpos_datarecorder.append(sensor_type, data)
            elif sensor_type == "VISO":
                self.viso_datarecorder.append(sensor_type, data)
            else:
                self.logger.error(f"Unknown sensor type: {sensor_type}")
