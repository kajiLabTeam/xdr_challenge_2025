import numpy as np
import numpy.typing as npt
from typing import final
from scipy.spatial.transform import Rotation as R
from src.lib.params._params import Params
from src.type import EstimateResult, TimedPose
from src.lib.recorder import DataRecorderProtocol
from src.lib.recorder.gpos import GposData
from src.lib.recorder.uwbp import UwbPData
from src.lib.recorder.uwbt import UwbTData
from src.lib.utils._utils import Utils


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

        # UWBP, UWBT, GPOSデータの組み合わせを作成
        uwb_gpos_data_dict: dict[
            str, list[tuple[float, UwbPData, UwbTData, GposData]]
        ] = {}
        for uwbp in uwbp_data:
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

            tag_id = uwbp["tag_id"]
            if tag_id not in uwb_gpos_data_dict:
                uwb_gpos_data_dict[tag_id] = []

            accuracy = self._uwb_calc_accuracy(uwbp, nearest_uwbt)
            uwb_gpos_data_dict[tag_id].append(
                (accuracy, uwbp, nearest_uwbt, nearest_gpos)
            )

        # 最も信頼度の高いタグを選択
        selected_tag_data = max(
            uwb_gpos_data_dict.values(),
            key=lambda x: float(np.mean([d[0] for d in x])),
            default=None,
        )
        if selected_tag_data is None:
            return (TimedPose(0, 0, 0, 0, 0), 0.0)

        # 信頼度の加重平均で位置を推定
        positions = np.array(
            [self._uwb_to_global_pos_by_uwbp(d[3], d[1]) for d in selected_tag_data]
        )
        accuracies = np.array([d[0] for d in selected_tag_data])
        weighted_position = np.sum(
            positions * accuracies.reshape(-1, 1), axis=0
        ) / np.sum(accuracies)

        pose = TimedPose(
            x=weighted_position[0],
            y=weighted_position[1],
            z=weighted_position[2],
            yaw=0,  # TODO
            timestamp=self.timestamp,
        )

        return (pose, accuracies.mean())

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
        if direction_norm < 0:
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
