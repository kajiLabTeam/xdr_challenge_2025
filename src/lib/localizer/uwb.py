import numpy as np
import numpy.typing as npt
from typing import final, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from src.lib.params._params import Params
from src.type import EstimateResult, Position
from src.lib.recorder import DataRecorderProtocol
from src.lib.recorder.gpos import GposData
from src.lib.recorder.uwbp import UwbPData
from src.lib.recorder.uwbt import UwbTData
from src.lib.recorder.ahrs import AhrsData
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

    @final
    def estimate_device_orientation(self) -> Optional[Tuple[R, float]]:
        """
        AHRSとUWBP/UWBTデータから端末の絶対的な向きを推定
        x軸正方向が0°となるようにヨー角を設定

        Returns:
            Tuple[Rotation, confidence]: 端末の姿勢と信頼度 (0.0-1.0)
            推定できない場合はNone
        """
        # 必要なデータの取得
        ahrs_data = self.ahrs_datarecorder.last_appended_data
        uwbp_data = self.uwbp_datarecorder.last_appended_data
        uwbt_data = list(self.uwbt_datarecorder.data[-100:])
        gpos_data = self.gpos_datarecorder.last_appended_data

        if not ahrs_data or not uwbp_data or not uwbt_data or not gpos_data:
            return None

        # 最新のAHRSデータから重力ベースのピッチとロールを取得
        latest_ahrs = ahrs_data[-1]
        pitch_deg = latest_ahrs["pitch_x"]  # 重力ベースで信頼できる
        roll_deg = latest_ahrs["roll_y"]  # 重力ベースで信頼できる

        # UWBから絶対ヨー角を計算
        yaw_result = self._calculate_absolute_yaw_from_uwb(
            uwbp_data, uwbt_data, gpos_data
        )

        if yaw_result is None:
            # UWBでヨーが計算できない場合、AHRSのみを使用（信頼度低）
            yaw_deg = latest_ahrs["yaw_z"]
            confidence = 0.3  # 磁気干渉の可能性があるため低信頼度
        else:
            yaw_deg, confidence = yaw_result

        # オイラー角から回転行列を作成（x軸正方向が0°）
        # 注: scipy.spatial.transform.Rotationの'xyz'順序を使用
        rotation = R.from_euler("xyz", [pitch_deg, roll_deg, yaw_deg], degrees=True)

        return (rotation, confidence)

    @final
    def _calculate_absolute_yaw_from_uwb(
        self,
        uwbp_data: list[UwbPData],
        uwbt_data: list[UwbTData],
        gpos_data: list[GposData],
    ) -> Optional[Tuple[float, float]]:
        """
        UWBデータから絶対的なヨー角を計算
        x軸正方向を0°とする座標系

        Returns:
            Tuple[yaw_degrees, confidence]: ヨー角（度）と信頼度
            計算できない場合はNone
        """
        best_estimate = None
        best_confidence = 0.0

        # 各タグについて処理
        for uwbp in uwbp_data:
            tag_id = uwbp["tag_id"]

            # 対応するUWBTデータを探す
            matching_uwbt = None
            for uwbt in uwbt_data:
                if (
                    uwbt["tag_id"] == tag_id
                    and abs(uwbt["app_timestamp"] - uwbp["app_timestamp"]) < 0.1
                ):
                    matching_uwbt = uwbt
                    break

            if matching_uwbt is None:
                continue

            # 対応するGPOSデータを探す
            matching_gpos = None
            for gpos in gpos_data:
                if gpos["object_id"] == tag_id:
                    matching_gpos = gpos
                    break

            if matching_gpos is None:
                continue

            # このタグペアから端末のヨー角を推定
            yaw_estimate = self._estimate_yaw_from_single_tag(
                uwbp, matching_uwbt, matching_gpos
            )

            if yaw_estimate is not None:
                yaw_deg, conf = yaw_estimate
                if conf > best_confidence:
                    best_estimate = yaw_deg
                    best_confidence = conf

        if best_estimate is not None:
            return (best_estimate, best_confidence)

        return None

    @final
    def _estimate_yaw_from_single_tag(
        self, uwbp: UwbPData, uwbt: UwbTData, gpos: GposData
    ) -> Optional[Tuple[float, float]]:
        """
        単一タグから端末のヨー角を推定

        Returns:
            Tuple[yaw_degrees, confidence]: ヨー角と信頼度
        """
        # タグの姿勢を取得
        R_tag = R.from_quat(
            [gpos["quat_x"], gpos["quat_y"], gpos["quat_z"], gpos["quat_w"]]
        )

        # タグから見た端末方向（タグ座標系）
        azimuth_rad = np.radians(uwbt["aoa_azimuth"])
        elevation_rad = np.radians(uwbt["aoa_elevation"])

        # タグ座標系での端末方向ベクトル
        d_tag_to_device_local = np.array(
            [
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.sin(elevation_rad),
            ]
        )

        # ワールド座標系での端末方向（タグから見た）
        d_tag_to_device_world = R_tag.apply(d_tag_to_device_local)

        # 端末から見たタグ方向（端末座標系）
        d_device_to_tag = np.array(
            [uwbp["direction_vec_x"], uwbp["direction_vec_y"], uwbp["direction_vec_z"]]
        )

        # 正規化
        norm = np.linalg.norm(d_device_to_tag)
        if norm <= 0:
            return None
        d_device_to_tag = d_device_to_tag / norm

        # ワールド座標系でのタグ方向（端末から見た）は
        # タグから見た端末方向の逆
        d_device_to_tag_world = -d_tag_to_device_world

        # 端末のヨー角を計算
        # 端末座標系のy軸（前方）がワールド座標系でどの方向を向いているか
        # d_device_to_tag が端末座標系での方向
        # d_device_to_tag_world がワールド座標系での同じ方向

        # 端末座標系での角度
        device_angle = np.arctan2(d_device_to_tag[1], d_device_to_tag[0])

        # ワールド座標系での角度（x軸正方向が0）
        world_angle = np.arctan2(d_device_to_tag_world[1], d_device_to_tag_world[0])

        # ヨー角 = ワールド角度 - デバイス角度
        yaw_rad = world_angle - device_angle

        # -πからπの範囲に正規化
        yaw_rad = np.arctan2(np.sin(yaw_rad), np.cos(yaw_rad))

        # 度に変換
        yaw_deg = np.degrees(yaw_rad)

        # 信頼度の計算
        distance = uwbp["distance"]
        nlos = uwbt["nlos"]
        time_diff = abs(uwbp["app_timestamp"] - uwbt["app_timestamp"])

        # 距離による信頼度（近いほど高い）
        dist_conf = max(0.0, min(1.0, 2.0 / (1.0 + distance)))

        # 時間差による信頼度（同期が取れているほど高い）
        time_conf = max(0.0, min(1.0, 1.0 / (1.0 + time_diff * 10)))

        # NLOSによる信頼度
        nlos_conf = 0.5 if nlos else 1.0

        # 総合信頼度
        confidence = dist_conf * time_conf * nlos_conf

        return (yaw_deg, confidence)

    @final
    def estimate_device_quaternion(
        self,
    ) -> Optional[Tuple[npt.NDArray[np.float64], float]]:
        """
        端末の姿勢をクォータニオンとして取得
        ground truthと同じ形式で出力

        Returns:
            Tuple[quaternion(x,y,z,w), confidence]: クォータニオンと信頼度
            推定できない場合はNone
        """
        orientation_result = self.estimate_device_orientation()

        if orientation_result is None:
            return None

        rotation, confidence = orientation_result

        # scipy形式のクォータニオン(x, y, z, w)を取得
        quat = rotation.as_quat()

        return (quat, confidence)
