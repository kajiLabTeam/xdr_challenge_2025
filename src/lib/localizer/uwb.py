import numpy as np
import numpy.typing as npt
import logging
from typing import final, Optional, Tuple
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
        
        # 時刻同期の最大許容誤差（秒）
        max_time_diff = 0.5  # 500ms以内のデータのみ使用（一時的に緩和）
        
        for uwbp in uwbp_data:
            # 時刻差が許容範囲内のUWBTデータを検索
            valid_uwbt_data = [
                x for x in uwbt_data 
                if abs(x["app_timestamp"] - uwbp["app_timestamp"]) <= max_time_diff
            ]
            
            if not valid_uwbt_data:
                continue
                
            nearest_uwbt = min(
                valid_uwbt_data,
                key=lambda x: abs(x["app_timestamp"] - uwbp["app_timestamp"])
            )
            
            nearest_gpos = min(
                gpos_data,
                key=lambda x: abs(x["app_timestamp"] - uwbp["app_timestamp"]),
                default=None,
            )
            if nearest_gpos is None:
                continue

            tag_id = uwbp["tag_id"]
            if tag_id not in uwb_gpos_data_dict:
                uwb_gpos_data_dict[tag_id] = []

            accuracy = self._uwb_calc_accuracy(uwbp, nearest_uwbt)
            uwb_gpos_data_dict[tag_id].append(
                (accuracy, uwbp, nearest_uwbt, nearest_gpos)
            )

        # 最も信頼度の高いタグを選択
        if not uwb_gpos_data_dict:
            # データがない場合はデバッグログを出力してデフォルト値を返す
            logging.warning(f"No valid UWB data combinations found - UWBP:{len(uwbp_data)}, UWBT:{len(uwbt_data)}, GPOS:{len(gpos_data)}")
            time = 0.0
            pose = TimedPose(
                timestamp=time,
                x=0.0,
                y=0.0,
                z=0.0,
                yaw=0.0,
            )
            return (pose, 0.0)
        
        selected_tag_data = max(
            uwb_gpos_data_dict.values(), key=lambda x: float(np.mean([d[0] for d in x]))
        )

        # 信頼度の加重平均で位置を推定
        valid_data = []
        for d in selected_tag_data:
            pos = self._uwb_to_global_pos_by_uwbp(d[3], d[1])
            if pos is not None:
                valid_data.append((pos, d[0]))  # (position, accuracy)
        
        if not valid_data:
            # 有効なデータがない場合はデフォルト位置を返す
            weighted_position = np.array([0.0, 0.0, 0.0])
            mean_accuracy = 0.0
        else:
            positions = np.array([d[0] for d in valid_data])
            accuracies = np.array([d[1] for d in valid_data])
            weighted_position = np.sum(
                positions * accuracies.reshape(-1, 1), axis=0
            ) / np.sum(accuracies)
            mean_accuracy = np.mean(accuracies)

        # 複数タグからのヨー角推定（信頼度の高いものを採用）
        yaw_estimations = [self._estimate_yaw_from_single_tag(d[1], d[2], d[3]) for d in selected_tag_data]

        best_estimate = None
        best_confidence = 0.0
        valid_yaw_count = 0
        
        for yaw_estimate in yaw_estimations:
            if yaw_estimate is not None:
                valid_yaw_count += 1
                yaw_deg, conf = yaw_estimate   # yaw_estimate を (ヨー角, 信頼度) のタプルとして取り出す
                if conf > best_confidence:     # 今までの中で一番信頼度が高いかどうかを比較
                    best_estimate = yaw_deg    # 一番信頼度の高いヨー角を記録
                    best_confidence = conf     # そのときの信頼度も更新

        logging.warning(f"Valid yaw estimations: {valid_yaw_count}/{len(yaw_estimations)}")
        if best_estimate is not None:
            logging.warning(f"Best yaw estimate: {best_estimate:.1f}°, confidence: {best_confidence:.3f}")
            yaw_rad = np.radians(best_estimate)
        else:
            logging.warning("No valid yaw estimates found, using 0.0")
            yaw_rad = 0.0

        time = selected_tag_data[-1][1]["app_timestamp"]
        pose = TimedPose(
            timestamp=time,
            x=weighted_position[0],
            y=weighted_position[1],
            z=weighted_position[2],
            yaw=yaw_rad,
        )

        return (pose, float(mean_accuracy))

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

        # 正規化（非常に小さい値も許容）
        norm = np.linalg.norm(d_device_to_tag)
        if norm < 1e-8:  # より小さな閾値を使用
            return None
        d_device_to_tag = d_device_to_tag / norm

        # ワールド座標系でのタグ方向（端末から見た）は
        # タグから見た端末方向の逆
        d_device_to_tag_world = -d_tag_to_device_world

        # 端末のヨー角を計算
        # 端末座標系での方向ベクトル d_device_to_tag
        # ワールド座標系での同じ方向ベクトル d_device_to_tag_world
        # 端末の姿勢を R_device とすると: d_device_to_tag_world = R_device * d_device_to_tag
        
        # 端末座標系でのx軸正方向への角度
        device_angle = np.arctan2(d_device_to_tag[1], d_device_to_tag[0])

        # ワールド座標系でのx軸正方向への角度
        world_angle = np.arctan2(d_device_to_tag_world[1], d_device_to_tag_world[0])

        # 端末のyaw角 = ワールド座標での観測方向 - 端末座標での観測方向
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