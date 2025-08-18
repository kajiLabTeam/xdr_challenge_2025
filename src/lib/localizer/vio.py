from typing import final
import numpy as np
from src.lib.recorder import DataRecorderProtocol
from src.lib.recorder._orientation import QOrientationWithTimestamp
from src.lib.recorder.viso import VisoData
from src.lib.safelist._safelist import SafeList
from src.type import Position


class VIOLocalizer(DataRecorderProtocol):
    """
    VIO による位置推定のためのクラス
    """

    _viso_init_direction: float | None = None
    _viso_orientations: SafeList[QOrientationWithTimestamp] = SafeList()

    _viso_tmp_positions: SafeList[Position] = SafeList()

    @final
    def estimate_vio(self) -> Position | None:
        """
        VIO による位置推定を行うメソッド
        Returns:
            Position: VIO による推定位置
        """
        try:
            viso_last_data = self.viso_datarecorder.last_appended_data[-1]
            return self._vio_to_global_position(viso_last_data)
        except IndexError:
            self.logger.debug("VISO データがありません")
            return None

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
    def _vio_initialize_direction(self, threshold: float = 6.0) -> float | None:
        """
        VISO の初期方向が設定されていない場合、 プロクラステス分析で初期方向を設定する
        Args:
            threshold (float): 初期方向を設定するための移動距離の閾値
        Returns:
            float: VISO の初期方向(ラジアン)
        """
        # 初期方向が設定されている場合
        if self._viso_init_direction is not None:
            return self._viso_init_direction

        gpos_data = self.gpos_datarecorder.data
        viso_data = self.viso_datarecorder.data
        first_gpos_data = gpos_data[0]
        last_gpos_data = gpos_data[-1]

        # 移動距離
        distance = np.linalg.norm(
            np.array([last_gpos_data["location_x"], last_gpos_data["location_y"]])
            - np.array([first_gpos_data["location_x"], first_gpos_data["location_y"]])
        )

        # {threshold}m 以上移動していない場合
        if distance < threshold:
            return None

        # 角度を計算
        angle_gpos = np.atan2(
            last_gpos_data["location_y"] - first_gpos_data["location_y"],
            last_gpos_data["location_x"] - first_gpos_data["location_x"],
        )
        angle_viso = np.atan2(
            viso_data[-1]["location_y"] - viso_data[0]["location_y"],
            viso_data[-1]["location_x"] - viso_data[0]["location_x"],
        )
        angle_rad = angle_gpos - angle_viso
        angle_deg = np.degrees(angle_rad)

        # 初期方向を設定
        self._viso_init_direction = angle_rad
        self.logger.info(
            f"VISO の初期方向を設定: {angle_rad:.1f}rad ({angle_deg:.1f}deg)"
        )

        return angle_rad
