import numpy as np
import numpy.typing as npt
from typing import final
from scipy.spatial.transform import Rotation as R
from src.lib.params._params import Params
from src.type import EstimateResult, Position
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
        position: Position,
        accuracy: float,
        distance: float,
        is_nlos: bool = True,
    ):
        self.tag_id = tag_id
        self.position = position
        self.accuracy = accuracy
        self.distance = distance
        self.is_nlos = is_nlos  # Non-Line of Sight フラグ


class UWBLocalizer(DataRecorderProtocol):
    """
    UWB による位置推定のためのクラス
    各タグごとに個別の推定軌跡を作成し、保持する。
    """

    def __init__(self) -> None:
        self.tag_priority = ("3637RLJ", "3636DWF", "3583WAA")

    @final
    def estimate_uwb(self) -> EstimateResult:
        """
        UWB による位置推定を行うメソッド
        """

        uwbp_data = self.uwbp_datarecorder.last_appended_data
        uwbt_data = self.uwbt_datarecorder.data[-100:]
        gpos_data = self.gpos_datarecorder.last_appended_data

        uwb_data_tag_dict: dict[str, list[tuple[np.ndarray, float, float]]] = {}

        for uwbp in uwbp_data:
            current_tag_id = uwbp["tag_id"]

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

            if current_tag_id not in uwb_data_tag_dict:
                uwb_data_tag_dict[current_tag_id] = []

            uwb_data_tag_dict[current_tag_id].append(
                (
                    pos,
                    accuracy,
                    uwbp["distance"],
                )
            )

        if len(uwb_data_tag_dict) == 0:
            return (Position(0, 0, 0), 0.0)

        # 各タグごとの推定位置を計算して保存
        tag_estimates: list[TagEstimate] = []
        for tag_id, pos_acc_list in uwb_data_tag_dict.items():
            if len(pos_acc_list) == 0:
                continue

            positions = np.array([p[0] for p in pos_acc_list])
            accuracies = np.array([p[1] for p in pos_acc_list])

            weighted_position = np.sum(
                positions * accuracies.reshape(-1, 1), axis=0
            ) / np.sum(accuracies)

            # 最新の推定を保存
            mean_distance = float(np.mean([p[2] for p in pos_acc_list]))

            tag_estimates.append(
                TagEstimate(
                    tag_id=tag_id,
                    position=Position(
                        float(weighted_position[0]),
                        float(weighted_position[1]),
                        float(weighted_position[2]),
                    ),
                    accuracy=accuracies.mean(),
                    distance=mean_distance,
                    is_nlos=False,
                )
            )

        # 最も信頼度の高いタグを選択
        selected_tag = max(tag_estimates, key=lambda t: t.accuracy, default=None)
        if selected_tag is None:
            return (Position(0, 0, 0), 0.0)

        return (
            selected_tag.position,
            selected_tag.accuracy,
        )

    @final
    def leave_ai_suitcase_time_first(self) -> float:
        """
        AIスーツケースから離れた最初の時間を取得
        初期進行方向推定で正解軌跡として使用できる終了時間
        """
        return self.uwbp_datarecorder.last_appended_data["app_timestamp"]

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
        信頼度を計算する
        """
        time_diff = abs(uwbp["app_timestamp"] - uwbt["app_timestamp"])
        time_diff_accuracy = Utils.sigmoid(time_diff, 5, 0.5)
        distance_accuracy = Utils.sigmoid(uwbp["distance"], 4, 2.0)
        los_accuracy = 1.0 if not uwbt["nlos"] else Params.uwb_nlos_factor()

        return time_diff_accuracy * distance_accuracy * los_accuracy

    @final
    def _uwb_cal_distance_accuracy(self, distance: float) -> float:
        """
        距離に基づく信頼度を計算

        Args:
            distance: UWB測定距離(m)

        Returns:
            0.0〜1.0の信頼度
        """
        return max(0.0, min(1.0, -0.2 * distance + 1))
