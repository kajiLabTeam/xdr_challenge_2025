from typing import cast, final
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from src.lib.recorder import DataRecorderProtocol
from src.lib.recorder._orientation import QOrientationWithTimestamp
from src.lib.recorder.viso import VisoData
from src.lib.safelist._safelist import SafeList
from src.lib.utils._utils import Utils
from src.type import EstimateResult, Position, QOrientation, TimedPose


class VIOLocalizer(DataRecorderProtocol):
    """
    VIO による位置推定のためのクラス
    """

    def __init__(self) -> None:
        self._viso_init_direction: float | None = None
        self._viso_orientations: SafeList[QOrientationWithTimestamp] = SafeList()
        self._viso_tmp_positions: SafeList[Position] = SafeList()
        self._vio_leave_timestamp: float | None = None

    @final
    def estimate_vio(self) -> EstimateResult:
        """
        VIO による位置推定を行うメソッド
        Returns:
            Position: VIO による推定位置
        """
        try:
            viso_last_data = self.viso_datarecorder.last_appended_data[-1]
            pose = self._vio_to_global_position(viso_last_data)
            if pose is None:
                return (TimedPose(0, 0, 0, 0, self.timestamp), 0.0)
            return (pose, 1.0)
        except IndexError:
            self.logger.debug("VISO データがありません")
            return (TimedPose(0, 0, 0, 0, self.timestamp), 0.0)

    @final
    def estimate_vio_orientations(self) -> SafeList[QOrientationWithTimestamp]:
        """
        VIO の姿勢を取得するメソッド
        Returns:
            SafeList[QOrientationWithTimestamp]: VIO の姿勢データ
        """
        orientations = self.viso_datarecorder.last_appended_orientations()
        self._viso_orientations.extend(*orientations)

        return orientations

    @final
    def _vio_to_global_position(self, data: VisoData) -> TimedPose | None:
        """
        グローバル位置を取得するメソッド
        初期位置・初期方向が取得できない場合は、None を返す

        Returns:
            Position: VISO のグローバル位置
        """
        first_gpos_data = self.gpos_datarecorder.first_data
        first_viso_data = self.viso_datarecorder.first_data
        init_dir = self._vio_initialize_direction()

        if first_gpos_data is None or first_viso_data is None or init_dir is None:
            return None

        pos = Position(
            x=data["location_x"] - first_viso_data["location_x"],
            y=data["location_y"] - first_viso_data["location_y"],
            z=data["location_z"] - first_viso_data["location_z"],
        )

        # 初期方向を考慮して回転
        rotated_x = pos.x * np.cos(init_dir) - pos.y * np.sin(init_dir)
        rotated_y = pos.x * np.sin(init_dir) + pos.y * np.cos(init_dir)
        yaw_original = Utils.quaternion_to_yaw(
            data["quat_w"],
            data["quat_x"],
            data["quat_y"],
            data["quat_z"],
        )
        yaw = (
            cast(float, yaw_original) - self._vio_switched_original_yaw
        ) + self._vio_switched_yaw

        return TimedPose(
            x=rotated_x + first_gpos_data["location_x"],
            y=rotated_y + first_gpos_data["location_y"],
            z=pos.z + first_gpos_data["location_z"],
            yaw=yaw,
            timestamp=self.timestamp,
        )

    @final
    def _vio_initialize_direction(self) -> float | None:
        """
        VISO の初期方向が設定されていない場合、 プロクラステス分析で初期方向を設定する
        Args:
            threshold (float): 初期方向を設定するための移動距離の閾値
        Returns:
            float: VISO の初期方向(ラジアン)
        """
        if hasattr(self, "_viso_init_direction"):
            return self._viso_init_direction

        viso_df = self.viso_datarecorder.df
        gpos_df = self.gpos_datarecorder.df
        timestamp_max = self._vio_get_uwb_leave_timestamp(2.0)

        df = pd.merge_asof(
            viso_df,
            gpos_df,
            on="app_timestamp",
            direction="nearest",
            suffixes=("_viso", "_gpos"),
        )

        if timestamp_max is not None:
            self.logger.info(f"timestamp_max: {timestamp_max}, {len(df)}")
            viso_df = viso_df[viso_df["app_timestamp"] <= timestamp_max]
            gpos_df = gpos_df[gpos_df["app_timestamp"] <= timestamp_max]

        V = df[["location_x_viso", "location_y_viso"]].to_numpy()
        G = df[["location_x_gpos", "location_y_gpos"]].to_numpy()

        R, _ = orthogonal_procrustes(V, G)
        # -180° ~ 180° に正規化
        angle_rad = np.arctan2(R[1, 0], R[0, 0]) % (2 * np.pi) - np.pi

        vio_last_data = self.viso_datarecorder.data[-1]
        x = vio_last_data["location_x"]
        y = vio_last_data["location_y"]

        is_inverted = y < x  # y=x の直線より下の場合は逆向きとみなす
        if is_inverted and abs(angle_rad) < np.pi / 2:
            angle_rad = (angle_rad + np.pi) % (2 * np.pi)

        angle_deg = np.degrees(angle_rad)

        if timestamp_max is not None:
            self._viso_init_direction = angle_rad

        return angle_rad

    @final
    def switch_to_vio(self, pose: TimedPose) -> None:
        """
        PDR切り替え時の初期設定を行う
        """
        last_data = self.viso_datarecorder.data[-1]
        original_yaw = Utils.quaternion_to_yaw(
            last_data["quat_w"],
            last_data["quat_x"],
            last_data["quat_y"],
            last_data["quat_z"],
        )
        self._vio_switched_original_yaw = cast(float, original_yaw)
        self._vio_switched_yaw = pose.yaw

    @final
    def _vio_get_uwb_leave_timestamp(self, threshold: float = 2.0) -> float | None:
        """
        UWBがAIスーツケースから threshold[m]以上離れた最初のタイムスタンプを取得する
        """
        if hasattr(self, "_vio_leave_timestamp"):
            return self._vio_leave_timestamp

        uwbt_data = self.uwbt_datarecorder.last_appended_data

        if len(uwbt_data) == 0:
            return None

        last_data = uwbt_data[-1]
        if last_data["distance"] < threshold:
            return None

        self._vio_leave_timestamp = last_data["app_timestamp"]
        return last_data["app_timestamp"]
