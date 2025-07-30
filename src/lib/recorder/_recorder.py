from logging import Logger
from typing import Protocol, final
from src.type import SensorData
from ._position import PositionDataRecorder, PositionDataRecorderProtocol
from .acce import AcceDataRecorder
from .ahrs import AhrsDataRecorder
from .gpos import GposDataRecorder
from .gyro import GyroDataRecorder
from .magn import MagnDataRecorder
from .uwbp import UwbPDataRecorder
from .uwbt import UwbTDataRecorder
from .viso import VisoDataRecorder


class DataRecorderProtocol(Protocol, PositionDataRecorderProtocol):
    trial_id: str
    logger: Logger
    acc_datarecorder: AcceDataRecorder
    gyro_datarecorder: GyroDataRecorder
    magn_datarecorder: MagnDataRecorder
    ahrs_datarecorder: AhrsDataRecorder
    uwbp_datarecorder: UwbPDataRecorder
    uwbt_datarecorder: UwbTDataRecorder
    gpos_datarecorder: GposDataRecorder
    viso_datarecorder: VisoDataRecorder

    def set_sensor_data(self, sensor_data: SensorData) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def clear_last_appended_data(self) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")


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

    @final
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

    @final
    def clear_last_appended_data(self) -> None:
        """
        最後に追加されたデータをクリアする
        last_appended_data が空になり、再度 append すると新しいデータが追加されます
        """
        self.acc_datarecorder.clear_last_appended_data()
        self.gyro_datarecorder.clear_last_appended_data()
        self.magn_datarecorder.clear_last_appended_data()
        self.ahrs_datarecorder.clear_last_appended_data()
        self.uwbp_datarecorder.clear_last_appended_data()
        self.uwbt_datarecorder.clear_last_appended_data()
        self.gpos_datarecorder.clear_last_appended_data()
        self.viso_datarecorder.clear_last_appended_data()
