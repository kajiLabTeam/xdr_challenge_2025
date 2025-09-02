import numpy as np
import numpy.typing as npt
from typing import final
from scipy.spatial.transform import Rotation as R
from src.lib.params._params import Params
from src.type import Position
from src.lib.recorder import DataRecorderProtocol
from src.lib.recorder.gpos import GposData
from src.lib.recorder.uwbp import UwbPData
from src.lib.recorder.uwbt import UwbTData
from src.lib.utils._utils import Utils


class TagEstimate:
    """各タグからの推定結果を格納するクラス"""

    def __init__(
        self,
        tag_id: str,
        position: np.ndarray,
        confidence: float,
        distance: float,
        method: str,
        is_nlos: bool = True,
    ):
        self.tag_id = tag_id
        self.position = position
        self.confidence = confidence
        self.distance = distance
        self.method = method  # 'UWBT' or 'UWBP'
        self.is_nlos = is_nlos  # Non-Line of Sight フラグ


class UWBLocalizer(DataRecorderProtocol):
    """
    UWB による位置推定のためのクラス

    各タグごとに個別の推定軌跡を作成し、保持する。
    """

    def __init__(self) -> None:
        self.tag_trajectories: dict[str, list[Position]] = {}
        self.tag_estimates: dict[str, TagEstimate] = {}

    @final
    def estimate_uwb(self, tag_id: str | None = None) -> Position | None:
        """特定のタグIDまたは全タグの統合推定を返す"""
        # 属性が存在しない場合は初期化
        if not hasattr(self, 'tag_trajectories'):
            self.tag_trajectories = {}
        if not hasattr(self, 'tag_estimates'):
            self.tag_estimates = {}
            
        uwbp_data = self.uwbp_datarecorder.last_appended_data
        uwbt_data = self.uwbt_datarecorder.data[-100:]
        gpos_data = self.gpos_datarecorder.last_appended_data

        tag_positions: dict[str, list[tuple[np.ndarray, float]]] = {}
        
        for uwbp in uwbp_data:
            current_tag_id = uwbp.get("tag_id", "unknown")
            
            if tag_id is not None and current_tag_id != tag_id:
                continue
            
            nearest_uwbt = min(
                uwbt_data,
                key=lambda x: abs(x["app_timestamp"] - uwbp["app_timestamp"]),
                default=None,
            )
            nearest_gpos = min(
                gpos_data,
                key=lambda x: abs(x["app_timestamp"] - uwbp["app_timestamp"]),
                default=None,
            )
            if nearest_uwbt is None or nearest_gpos is None:
                continue

            accuracy = self._uwb_calc_accuracy(uwbp, nearest_uwbt)
            pos = self._uwb_to_global_pos_by_uwbp(nearest_gpos, uwbp)

            if pos is None:
                continue

            if current_tag_id not in tag_positions:
                tag_positions[current_tag_id] = []
            
            tag_positions[current_tag_id].append(
                (
                    pos,
                    accuracy,
                )
            )

        if len(tag_positions) == 0:
            return None

        # 各タグごとの推定位置を計算して保存
        for tid, pos_acc_list in tag_positions.items():
            if len(pos_acc_list) == 0:
                continue
            
            positions = np.array([p[0] for p in pos_acc_list])
            accuracies = np.array([p[1] for p in pos_acc_list])
            
            weighted_position = np.sum(
                positions * accuracies.reshape(-1, 1), axis=0
            ) / np.sum(accuracies)
            
            estimated_pos = Position(
                x=float(weighted_position[0]),
                y=float(weighted_position[1]),
                z=float(weighted_position[2]),
            )
            
            # タグごとの軌跡に追加
            if tid not in self.tag_trajectories:
                self.tag_trajectories[tid] = []
            self.tag_trajectories[tid].append(estimated_pos)
            
            # 最新の推定を保存
            self.tag_estimates[tid] = TagEstimate(
                tag_id=tid,
                position=weighted_position,
                confidence=float(np.mean(accuracies)),
                distance=float(np.mean([uwbp["distance"] for uwbp in uwbp_data if uwbp.get("tag_id") == tid])),
                method="UWBP",
                is_nlos=False
            )
        
        # 特定のタグIDが指定されている場合はその推定のみ返す
        if tag_id is not None:
            if tag_id in self.tag_estimates:
                est = self.tag_estimates[tag_id]
                return Position(
                    x=float(est.position[0]),
                    y=float(est.position[1]),
                    z=float(est.position[2]),
                )
            return None
        
        # tag_idが指定されていない場合は全タグの統合推定を返す
        all_positions = []
        all_accuracies = []
        for tid, pos_acc_list in tag_positions.items():
            all_positions.extend([p[0] for p in pos_acc_list])
            all_accuracies.extend([p[1] for p in pos_acc_list])
        
        if len(all_positions) == 0:
            return None
        
        positions = np.array(all_positions)
        accuracies = np.array(all_accuracies)
        
        weighted_position = np.sum(
            positions * accuracies.reshape(-1, 1), axis=0
        ) / np.sum(accuracies)

        return Position(
            x=float(weighted_position[0]),
            y=float(weighted_position[1]),
            z=float(weighted_position[2]),
        )

    @final
    def _uwb_to_global_pos_by_uwbp(
        self, gpos: GposData, uwbp: UwbPData
    ) -> npt.NDArray[np.float64] | None:
        """
        GPOSとUWBPデータを統合して位置情報を生成する
        """
        direction_vec = np.array(
            [
                uwbp["direction_vec_x"],
                uwbp["direction_vec_y"],
                uwbp["direction_vec_z"],
            ]
        )

        # 方向ベクトルを正規化
        direction_norm = np.linalg.norm(direction_vec)
        if direction_norm <= 0:
            return None

        direction_vec = direction_vec / direction_norm

        local_point = uwbp["distance"] * direction_vec
        location = np.array(
            [
                gpos["location_x"],
                gpos["location_y"],
                gpos["location_z"],
            ]
        )
        quat = np.array(
            [
                gpos["quat_x"],
                gpos["quat_y"],
                gpos["quat_z"],
                gpos["quat_w"],
            ]
        )
        quat = quat / np.linalg.norm(quat)
        R_tag = R.from_quat(quat)
        world_point = R_tag.apply(local_point) + location

        return world_point

    @final
    def _uwb_to_global_pos_by_uwbt(
        self, gpos: GposData, uwbt: UwbTData
    ) -> npt.NDArray[np.float64] | None:
        """
        GPOSとUWBTデータを統合して位置情報を生成する
        """
        distance = uwbt["distance"]
        azimuth_rad = np.radians(uwbt["aoa_azimuth"])
        elevation_rad = np.radians(uwbt["aoa_elevation"])

        # ローカル座標系での相対位置
        x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        z = distance * np.sin(elevation_rad)
        local_point = np.array([x, y, z])

        location = np.array(
            [
                gpos["location_x"],
                gpos["location_y"],
                gpos["location_z"],
            ]
        )
        quat = np.array(
            [
                gpos["quat_x"],
                gpos["quat_y"],
                gpos["quat_z"],
                gpos["quat_w"],
            ]
        )
        quat = quat / np.linalg.norm(quat)

        # タグの姿勢で回転してワールド座標に変換
        R_tag = R.from_quat(quat)
        world_point = R_tag.apply(local_point) + location

        return world_point

    @final
    def _uwb_calc_accuracy(self, uwbp: UwbPData, uwbt: UwbTData) -> float:
        """
        確信度を計算する
        """
        time_diff = abs(uwbp["app_timestamp"] - uwbt["app_timestamp"])
        time_diff_accuracy = Utils.sigmoid(time_diff, 5, 0.5)
        distance_accuracy = Utils.sigmoid(uwbp["distance"], 4, 2.0)
        los_accuracy = 1.0 if not uwbt["nlos"] else Params.uwb_nlos_factor()

        return time_diff_accuracy * distance_accuracy * los_accuracy
    
    @final
    def _get_tag_trajectory(self, tag_id: str) -> list[Position] | None:
        """特定のタグIDの軌跡を取得"""
        if not hasattr(self, 'tag_trajectories'):
            return None
        return self.tag_trajectories.get(tag_id)
    
    @final
    def _get_all_tag_trajectories(self) -> dict[str, list[Position]]:
        """全てのタグの軌跡を取得"""
        if not hasattr(self, 'tag_trajectories'):
            return {}
        return self.tag_trajectories
    
    @final
    def _get_tag_estimate(self, tag_id: str) -> TagEstimate | None:
        """特定のタグの最新推定を取得"""
        if not hasattr(self, 'tag_estimates'):
            return None
        return self.tag_estimates.get(tag_id)
    
    @final
    def _clear_tag_trajectory(self, tag_id: str | None = None) -> None:
        """タグの軌跡をクリア"""
        if not hasattr(self, 'tag_trajectories'):
            self.tag_trajectories = {}
        if not hasattr(self, 'tag_estimates'):
            self.tag_estimates = {}
            
        if tag_id is None:
            self.tag_trajectories.clear()
            self.tag_estimates.clear()
        elif tag_id in self.tag_trajectories:
            del self.tag_trajectories[tag_id]
            if tag_id in self.tag_estimates:
                del self.tag_estimates[tag_id]
