from typing import TypedDict, cast
import numpy as np
from src.lib.params import Params
from ._base import BaseDataRecorder
from scipy.spatial.transform import Rotation as R


class GposData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    object_id: str
    location_x: float
    location_y: float
    location_z: float
    quat_x: float
    quat_y: float
    quat_z: float
    quat_w: float


class GposDataRecorder(BaseDataRecorder[GposData]):
    key = "GPOS"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "object_id": str,
        "location_x": float,
        "location_y": float,
        "location_z": float,
        "quat_x": float,
        "quat_y": float,
        "quat_z": float,
        "quat_w": float,
    }

    @property
    def yaw(self) -> float:
        """
        GPOSの進行方向からyaw角を返す
        """
        last_data = self.last_appended_data
        if len(last_data) < 1:
            return 0.0

        object_id = last_data[-1]["object_id"]
        rot = R.from_quat(
            [
                last_data[-1]["quat_x"],
                last_data[-1]["quat_y"],
                last_data[-1]["quat_z"],
                last_data[-1]["quat_w"],
            ]
        )

        # 参考: https://github.com/PDR-benchmark-standardization-committee/xdr-challenge-2025-examples/blob/main/competition_coordinates.pdf

        # x→z, y→y, z→-x に変換
        if object_id == "3636DWF":  # tag1
            R_swap = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        # x→-y, y→-x, z→-z に変換
        elif object_id == "3637RLJ":  # tag2
            R_swap = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

        # x→y, y→x, z→-z に変換
        elif object_id == "3583WAA":  # tag3
            R_swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        # base_link 等
        else:
            R_swap = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        R_gpos = rot.as_matrix() @ R_swap
        yaw = np.arctan2(R_gpos[1, 0], R_gpos[0, 0])
        return cast(float, yaw)

    @property
    def adjusted_yaw(self) -> float:
        """
        GPOSの進行方向からyaw角を返す (0~2πに正規化)
        """
        return self.yaw + Params.yaw_adjust()
