from typing import final
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from src.lib.recorder import DataRecorderProtocol
from src.lib.recorder._orientation import QOrientationWithTimestamp
from src.lib.recorder.viso import VisoData
from src.lib.safelist._safelist import SafeList
from src.type import EstimateResult, Position


class VIOLocalizer(DataRecorderProtocol):
    """
    VIO による位置推定のためのクラス
    """

    def __init__(self) -> None:
        self._viso_init_direction: float | None = None
        self._viso_orientations: SafeList[QOrientationWithTimestamp] = SafeList()
        self._viso_tmp_positions: SafeList[Position] = SafeList()

    @final
    def estimate_vio(self) -> EstimateResult:
        """
        VIO による位置推定を行うメソッド
        Returns:
            Position: VIO による推定位置
        """
        try:
            viso_last_data = self.viso_datarecorder.last_appended_data[-1]
            pos = self._vio_to_global_position(viso_last_data)
            if pos is None:
                return (Position(0, 0, 0), 0.0)
            return (pos, 1.0)
        except IndexError:
            self.logger.debug("VISO データがありません")
            return (Position(0, 0, 0), 0.0)

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
    def _vio_to_global_position(self, data: VisoData) -> Position | None:
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

        return Position(
            x=rotated_x + first_gpos_data["location_x"],
            y=rotated_y + first_gpos_data["location_y"],
            z=pos.z + first_gpos_data["location_z"],
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
        viso_df = self.viso_datarecorder.df
        gpos_df = self.gpos_datarecorder.df
        df = pd.merge_asof(
            viso_df,
            gpos_df,
            on="app_timestamp",
            direction="nearest",
            suffixes=("_viso", "_gpos"),
        )

        if len(df) > 4000:
            return self._viso_init_direction

        G = df[["location_x_gpos", "location_y_gpos"]].to_numpy()
        V = df[["location_x_viso", "location_y_viso"]].to_numpy()

        R, _ = orthogonal_procrustes(V, G)
        vec_X = G[-1] - V[0]
        vec_Y = G[-1] - G[0]
        cos_angle = np.dot(vec_X, vec_Y) / (np.linalg.norm(vec_X) * np.linalg.norm(vec_Y))

        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        # 初期方向を設定
        self._viso_init_direction = angle_rad
        self.logger.info(
            f"VISO の初期方向を設定: {angle_rad:.1f}rad ({angle_deg:.1f}deg)"
        )

        return angle_rad
