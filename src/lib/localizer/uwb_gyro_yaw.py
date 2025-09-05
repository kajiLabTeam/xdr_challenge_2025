import numpy as np
import numpy.typing as npt
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
        
        # コンプリメンタリフィルタ用の状態変数
        self.previous_yaw: Optional[float] = None  # 前回のヨー角（度）
        self.previous_timestamp: Optional[float] = None  # 前回のタイムスタンプ
        self.complementary_filter_alpha = 0.95  # ジャイロの重み（0.95 = 95%ジャイロ、5%UWB）
        

    @final
    def estimate_uwb(self) -> EstimateResult:
        """
        UWB による位置推定を行うメソッド
        """

        uwbp_data = self.uwbp_datarecorder.last_appended_data
        uwbt_data = self.uwbt_datarecorder.data[-100:]
        gpos_data = self.gpos_datarecorder.last_appended_data
        ahrs_data = self.ahrs_datarecorder.last_appended_data

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
        if not uwb_gpos_data_dict:
            # データがない場合は最後の位置を返す
            return (self.last_pose, 0.1)
        
        selected_tag_data = max(
            uwb_gpos_data_dict.values(), key=lambda x: float(np.mean([d[0] for d in x]))
        )

        # 信頼度の加重平均で位置を推定
        positions_list = []
        accuracies_list = []
        for d in selected_tag_data:
            pos = self._uwb_to_global_pos_by_uwbp(d[3], d[1])
            if pos is not None:
                positions_list.append(pos)
                accuracies_list.append(d[0])
        
        if not positions_list:
            # 有効な位置が取得できなかった場合、最後の位置を返す
            return (self.last_pose, 0.1)
        
        positions = np.array(positions_list)
        accuracies = np.array(accuracies_list)
        weighted_position = np.sum(
            positions * accuracies.reshape(-1, 1), axis=0
        ) / np.sum(accuracies)

        # 現在のタイムスタンプを取得
        time = selected_tag_data[-1][1]["app_timestamp"]
        
        # UWBからヨー角を推定（複数タグから最も信頼度の高いものを選択）
        uwb_yaw = None
        uwb_yaw_confidence = 0.0
        
        for accuracy, uwbp, uwbt, gpos in selected_tag_data:
            yaw_estimate = self._estimate_yaw_from_single_tag(uwbp, uwbt, gpos)
            if yaw_estimate is not None:
                yaw_deg, conf = yaw_estimate
                if conf > uwb_yaw_confidence:
                    uwb_yaw = yaw_deg
                    uwb_yaw_confidence = conf
        
        # コンプリメンタリフィルタでヨー角を融合
        # UWBデータがない場合でもジャイロで補間される
        fused_yaw = self._apply_complementary_filter(
            uwb_yaw, 
            uwb_yaw_confidence,
            time
        )
        pose = TimedPose(
            timestamp=time,
            x=weighted_position[0],
            y=weighted_position[1],
            z=weighted_position[2],
            yaw=fused_yaw,
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
    def _get_gyro_yaw_rate(self, timestamp: float) -> Optional[float]:
        """
        ジャイロデータからヨー角速度（度/秒）を取得
        
        Args:
            timestamp: 現在のタイムスタンプ
            
        Returns:
            ヨー角速度（度/秒）、データがない場合はNone
        """
        gyro_data = self.gyro_datarecorder.last_appended_data
        if not gyro_data:
            return None
            
        # タイムスタンプに最も近いデータを取得
        closest_gyro = min(
            gyro_data,
            key=lambda x: abs(x["app_timestamp"] - timestamp),
            default=None
        )
        
        if closest_gyro is None:
            return None
            
        # z軸周りの角速度をラジアン/秒から度/秒に変換
        # 注: ジャイロデータは通常ラジアン/秒で提供される
        yaw_rate_deg_per_sec = np.degrees(closest_gyro["gyr_z"])
        
        return yaw_rate_deg_per_sec
    
    @final
    def _apply_complementary_filter(
        self,
        uwb_yaw: Optional[float],
        uwb_confidence: float,
        current_timestamp: float
    ) -> float:
        """
        コンプリメンタリフィルタでジャイロとUWBのヨー角を融合
        
        Args:
            uwb_yaw: UWBから推定したヨー角（度）、Noneの場合もあり
            uwb_confidence: UWBヨー角の信頼度 (0.0-1.0)
            current_timestamp: 現在のタイムスタンプ
            
        Returns:
            融合後のヨー角（度）
        """
        # 初回実行時
        if self.previous_yaw is None or self.previous_timestamp is None:
            if uwb_yaw is not None:
                self.previous_yaw = uwb_yaw
                self.previous_timestamp = current_timestamp
                return uwb_yaw
            else:
                # UWBデータもない場合は0度で初期化
                self.previous_yaw = 0.0
                self.previous_timestamp = current_timestamp
                return 0.0
        
        # 時間差を計算
        dt = current_timestamp - self.previous_timestamp
        if dt <= 0:
            return self.previous_yaw
        
        # ジャイロによる角度変化を計算
        gyro_rate = self._get_gyro_yaw_rate(current_timestamp)
        if gyro_rate is not None:
            gyro_delta = gyro_rate * dt
            gyro_yaw = self.previous_yaw + gyro_delta
        else:
            gyro_yaw = self.previous_yaw
        
        # UWBデータがある場合、コンプリメンタリフィルタで融合
        if uwb_yaw is not None:
            # 角度差を-180〜180度の範囲に正規化
            angle_diff = uwb_yaw - gyro_yaw
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
            
            # コンプリメンタリフィルタ適用
            fused_yaw = self.complementary_filter_alpha * gyro_yaw + (1.0 - self.complementary_filter_alpha) * uwb_yaw
        else:
            # UWBデータがない場合はジャイロのみ
            fused_yaw = gyro_yaw
        
        # -180〜180度の範囲に正規化
        while fused_yaw > 180:
            fused_yaw -= 360
        while fused_yaw < -180:
            fused_yaw += 360
        
        # 状態を更新
        self.previous_yaw = fused_yaw
        self.previous_timestamp = current_timestamp
        
        return fused_yaw

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
        # 球座標系 (r, azimuth, elevation) からカルテシアン座標系 (x, y, z) への変換
        # x軸: 東方向, y軸: 北方向, z軸: 上方向
        d_tag_to_device_local = np.array(
            [
                np.cos(elevation_rad) * np.cos(azimuth_rad),  # x = r * cos(el) * cos(az)
                np.cos(elevation_rad) * np.sin(azimuth_rad),  # y = r * cos(el) * sin(az)
                np.sin(elevation_rad),                        # z = r * sin(el)
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

        # 端末のyaw角を推定するため、以下の関係を利用：
        # 端末座標系の基準ベクトル（前方: [0, 1, 0]）が
        # ワールド座標系でどの方向を向いているかを求める
        
        # 端末座標系でのタグ方向をyaw回転で変換すると
        # ワールド座標系でのタグ方向になる
        # 回転行列: R_y(yaw) = [[cos(yaw), -sin(yaw), 0],
        #                       [sin(yaw),  cos(yaw), 0],
        #                       [0,         0,        1]]
        
        # d_device_to_tag_world = R_y(yaw) * d_device_to_tag
        # これを解くと:
        
        # XY平面での角度のみを考慮（Z成分は無視）
        device_xy = d_device_to_tag[:2]  # [x, y]
        world_xy = d_device_to_tag_world[:2]  # [x, y] in world
        
        # 端末座標系での方向角
        device_angle = np.arctan2(device_xy[1], device_xy[0])
        
        # ワールド座標系での方向角
        world_angle = np.arctan2(world_xy[1], world_xy[0])
        
        # yaw = world_angle - device_angle
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

