import numpy as np
import numpy.typing as npt
import logging
from typing import final, Optional, Tuple
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)
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
        self.yaw_history: list[float] = []  # yaw角の履歴（平滑化用）
        self.max_history_size = 5  # 履歴サイズ
        # 方向ベクトルの履歴管理
        self.direction_history: dict[str, list[npt.NDArray[np.float64]]] = {}
        self.max_direction_history_size = 3  # 方向ベクトル履歴サイズ
        # 過去の推定位置履歴（連続性チェック用）
        self.previous_positions: list[npt.NDArray[np.float64]] = []
        self.max_position_history_size = 5
        # デバイス向き補正
        self.device_orientation_matrix: Optional[npt.NDArray[np.float64]] = None
        self.gravity_calibrated = False

    @final
    def estimate_uwb(self) -> EstimateResult:
        """
        UWB による位置推定を行うメソッド
        """

        # デバイス向き補正の初期化（初回のみ）
        if not self.gravity_calibrated:
            self._calibrate_device_orientation()

        uwbp_data = self.uwbp_datarecorder.last_appended_data
        uwbt_data = self.uwbt_datarecorder.data[-100:]
        gpos_data = self.gpos_datarecorder.last_appended_data

        # UWBP, UWBT, GPOSデータの組み合わせを作成
        uwb_gpos_data_dict: dict[
            str, list[tuple[float, UwbPData, UwbTData, GposData]]
        ] = {}
        
        # 時刻同期の最大許容誤差（秒）
        max_time_diff = 0.1  # 100ms以内のデータのみ使用（厳格化）
        
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

            # UWBTデータに依存しない信頼度計算
            accuracy = self._uwb_calc_accuracy_direction_only(uwbp)
            # 連続性を考慮した信頼度を計算
            continuity_accuracy = self._calc_continuity_accuracy(uwbp, nearest_gpos)
            total_accuracy = accuracy * continuity_accuracy
            uwb_gpos_data_dict[tag_id].append(
                (total_accuracy, uwbp, nearest_uwbt, nearest_gpos)
            )

        # 最も信頼度の高いタグを選択
        if not uwb_gpos_data_dict:
            time = 0.0
            pose = TimedPose(
                timestamp=time,
                x=0.0,
                y=0.0,
                z=0.0,
                yaw=0.0,
            )
            return (pose, 0.0)
        
        # タグ別信頼度を分析・デバッグ出力
        tag_scores = {}
        for tag_id, data_list in uwb_gpos_data_dict.items():
            avg_score = float(np.mean([d[0] for d in data_list]))
            tag_scores[tag_id] = avg_score
        
        selected_tag_id = max(tag_scores.keys(), key=lambda k: tag_scores[k])
        selected_tag_data = uwb_gpos_data_dict[selected_tag_id]

        # 信頼度の加重平均で位置を推定
        valid_data = []
        for d in selected_tag_data:
            # 方向ベクトルの平滑化処理を適用
            smoothed_uwbp = self._smooth_direction_vector(d[1])
            if smoothed_uwbp is not None:
                pos = self._uwb_to_global_pos_by_uwbp(d[3], smoothed_uwbp)
                if pos is not None:
                    valid_data.append((pos, d[0]))  # (position, accuracy)
        
        # 統計的外れ値除去を適用
        valid_data = self._remove_statistical_outliers(valid_data)
        
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
        
        # 推定位置の履歴を更新
        self._update_position_history(weighted_position)

        # direction_vecのみを使用したヨー角推定（UWBTデータは使用しない）
        yaw_estimations = [self._estimate_yaw_from_direction_vec(d[1], d[3]) for d in selected_tag_data]

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

       
        if best_estimate is not None:
            # 角度のラップアラウンド問題を修正するため、連続性を考慮した平滑化
            if len(self.yaw_history) > 0:
                # 前回の値との差が180度以上なら、ラップアラウンドを修正
                last_yaw = self.yaw_history[-1]
                diff = best_estimate - last_yaw
                
                # 角度差を-180~180度の範囲に正規化
                while diff > 180:
                    best_estimate -= 360
                    diff = best_estimate - last_yaw
                while diff < -180:
                    best_estimate += 360
                    diff = best_estimate - last_yaw
            
            # 移動平均による平滑化
            self.yaw_history.append(best_estimate)
            if len(self.yaw_history) > self.max_history_size:
                self.yaw_history.pop(0)
            
            # 単純平均（連続性を保持済み）
            smoothed_yaw_deg = np.mean(self.yaw_history)
            yaw_rad = np.radians(smoothed_yaw_deg)
            
            # -π~πの範囲に正規化
            yaw_rad = np.arctan2(np.sin(yaw_rad), np.cos(yaw_rad))
        else:
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
        GPOSとUWBPデータを統合して位置情報を生成する（X軸=前方基準）
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

        # ローカル座標系での相対位置（X軸=前方基準）
        # azimuth=0度がX軸正方向（前方）を指すと仮定
        # X軸=前方, Y軸=右方, Z軸=上方
        x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)  # 前方成分
        y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)  # 右方成分  
        z = distance * np.sin(elevation_rad)                        # 上方成分
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
        # タグの姿勢を取得（他の箇所と同様に正規化）
        quat = np.array(
            [gpos["quat_x"], gpos["quat_y"], gpos["quat_z"], gpos["quat_w"]]
        )
        quat = quat / np.linalg.norm(quat)
        R_tag = R.from_quat(quat)

        # タグから見た端末方向（タグ座標系）
        azimuth_rad = np.radians(uwbt["aoa_azimuth"])
        elevation_rad = np.radians(uwbt["aoa_elevation"])

        # タグ座標系での端末方向ベクトル（X軸=前方基準に統一）
        # _uwb_to_global_pos_by_uwbtと同じ座標系に合わせる
        # x = cos(elevation) * cos(azimuth)  # 前方成分
        # y = cos(elevation) * sin(azimuth)  # 右方成分
        # z = sin(elevation)                 # 上方成分
        d_tag_to_device_local = np.array(
            [
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad),
            ]
        )

        # ワールド座標系での端末方向（タグから見た）
        d_tag_to_device_world = R_tag.apply(d_tag_to_device_local)

        # 端末から見たタグ方向（端末座標系）
        d_device_to_tag_raw = np.array(
            [uwbp["direction_vec_x"], uwbp["direction_vec_y"], uwbp["direction_vec_z"]]
        )

        # 正規化（非常に小さい値も許容）
        norm = np.linalg.norm(d_device_to_tag_raw)
        if norm < 1e-8:  # より小さな閾値を使用
            return None
        d_device_to_tag_raw = d_device_to_tag_raw / norm
        
        # デバイス向き補正を適用
        d_device_to_tag = self._apply_device_orientation_correction(d_device_to_tag_raw)

        # ワールド座標系でのタグ方向（端末から見た）は
        # タグから見た端末方向の逆
        d_device_to_tag_world = -d_tag_to_device_world

        # 端末のヨー角を計算
        # 座標系の定義：x軸=前向き（人が向いている方向）、y軸=右方
        # azimuth=0度はx軸正方向（前向き）を基準とする
        
        # ワールド座標系でのタグ方向（X軸=前向きから反時計回りで正）
        world_yaw_to_tag = np.arctan2(d_device_to_tag_world[1], d_device_to_tag_world[0])
        
        # 端末座標系でのタグ方向（端末のX軸=前方から反時計回りで正）
        # 端末座標系：x軸=前方, y軸=右方
        device_yaw_to_tag = np.arctan2(d_device_to_tag[1], d_device_to_tag[0])
        
        # 端末のワールド座標でのyaw角
        yaw_rad = world_yaw_to_tag - device_yaw_to_tag

        # -πからπの範囲に正規化（四分円エラーを防止）
        # より安定した正規化方法を使用
        if yaw_rad > np.pi:
            yaw_rad -= 2 * np.pi
        elif yaw_rad < -np.pi:
            yaw_rad += 2 * np.pi

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
    def _estimate_yaw_from_direction_vec(self, uwbp: UwbPData, gpos: GposData) -> Optional[Tuple[float, float]]:
        """
        direction_vecから直接yaw角を計算（座標変換を最小化）
        
        Args:
            uwbp: UWB proximity data
            gpos: GPOS data (現在は使用しないが署名の互換性のため残す)
        
        Returns:
            Tuple[yaw_degrees, confidence]: ヨー角と信頼度
        """
        # Note: gpos parameter is currently unused but kept for signature compatibility
        # direction_vecを取得
        direction_vec = np.array([
            uwbp["direction_vec_x"], 
            uwbp["direction_vec_y"], 
            uwbp["direction_vec_z"]
        ])
        
        # 正規化チェック
        norm = np.linalg.norm(direction_vec)
        if norm <= 0:
            return None
        
        # 正規化
        direction_vec = direction_vec / norm
        
        # AHRSベースの端末向き補正を適用
        corrected_direction_vec = self._apply_device_orientation_correction(direction_vec)
        
        # 単純なyaw角計算（XY平面での角度）
        # 補正後のdirection_vecを使用
        # 複数パターンを試して最適なものを選択
        yaw_candidates = self._calculate_yaw_candidates(corrected_direction_vec)
        
        # 履歴ベースで最も一貫性のある角度を選択
        yaw_deg = self._select_most_consistent_yaw(yaw_candidates)
        
        # -180度から180度の範囲に正規化
        yaw_deg = self._normalize_angle(yaw_deg)
        
        # 信頼度計算（シンプル化）
        distance = uwbp["distance"]
        
        # 距離による信頼度（近いほど高い、最大3m）
        dist_conf = max(0.1, min(1.0, (3.0 - distance) / 3.0)) if distance > 0 else 0.1
        
        # direction_vecの正規化度による信頼度
        norm_conf = min(1.0, float(norm))
        
        # Z成分による信頼度（水平に近いほど良い）
        z_conf = max(0.5, 1.0 - abs(direction_vec[2]) * 0.5)
        
        # 総合信頼度
        confidence = dist_conf * norm_conf * z_conf
        
        
        # Priority 2: 角度平滑化処理を適用
        smoothed_yaw = self._smooth_yaw_angle(yaw_deg, confidence)

        return (smoothed_yaw, confidence)

    @final
    def _calculate_yaw_candidates(self, direction_vec: npt.NDArray[np.float64]) -> list[Tuple[str, float]]:
        """
        複数の変換パターンでyaw角候補を計算
        
        Args:
            direction_vec: 補正済みの方向ベクトル
            
        Returns:
            List[(pattern_name, yaw_degrees)]: パターン名と対応するyaw角のリスト
        """
        candidates = []
        
        # パターン1: そのまま（タグ方向）
        yaw1 = np.degrees(np.arctan2(direction_vec[1], direction_vec[0]))
        candidates.append(("direct", yaw1))
        
        # パターン2: 反転（人の向き = タグと逆）
        yaw2 = np.degrees(np.arctan2(-direction_vec[1], -direction_vec[0]))
        candidates.append(("inverse", yaw2))
        
        # パターン3: 90度回転
        yaw3 = np.degrees(np.arctan2(-direction_vec[0], direction_vec[1]))
        candidates.append(("rotated_90", yaw3))
        
        # パターン4: -90度回転  
        yaw4 = np.degrees(np.arctan2(direction_vec[0], -direction_vec[1]))
        candidates.append(("rotated_-90", yaw4))
        
        return candidates
    
    @final
    def _select_most_consistent_yaw(self, candidates: list[Tuple[str, float]]) -> float:
        """
        履歴との一貫性を考慮して最適なyaw角を選択
        
        Args:
            candidates: yaw角候補のリスト
            
        Returns:
            最も一貫性のあるyaw角
        """
        if not self.yaw_history:
            # 履歴がない場合はinverse（人の向き）を選択
            for pattern_name, yaw in candidates:
                if pattern_name == "inverse":
                    logger.info(f"Initial pattern selected: {pattern_name}, yaw: {yaw:.1f}°")
                    return yaw
            return candidates[0][1]  # fallback
        
        # 最後のyaw角との差分を計算
        last_yaw = self.yaw_history[-1]
        best_candidate = None
        min_diff = float('inf')
        
        for pattern_name, yaw in candidates:
            diff = abs(self._angle_difference(last_yaw, yaw))
            if diff < min_diff:
                min_diff = diff
                best_candidate = (pattern_name, yaw)
        
        if best_candidate and min_diff > 90:  # 90度以上の差がある場合は警告
            logger.info(f"Large yaw change detected: {min_diff:.1f}° (pattern: {best_candidate[0]})")
        
        return best_candidate[1] if best_candidate else candidates[0][1]

    @final
    def _uwb_calc_accuracy_direction_only(self, uwbp: UwbPData) -> float:
        """
        direction_vecのみを使用した信頼度計算（UWBTデータ不使用）
        """
        # direction_vecの品質チェック
        direction_vec = np.array([uwbp["direction_vec_x"], uwbp["direction_vec_y"], uwbp["direction_vec_z"]])
        direction_norm = np.linalg.norm(direction_vec)
        
        # 正規化度による信頼度（1に近いほど良い）
        norm_accuracy = min(1.0, float(direction_norm)) if direction_norm > 0 else 0.0
        
        # 距離による信頼度
        distance_accuracy = Utils.sigmoid(uwbp["distance"], 4, 2.0)
        
        # direction_vecの成分バランスチェック
        # 極端に片寄った方向ベクトルは信頼度を下げる
        if direction_norm > 0:
            normalized_vec = direction_vec / direction_norm
            component_balance = 1.0 - np.max(np.abs(normalized_vec)) + 0.5  # 0.5-1.5の範囲
            component_balance = np.clip(component_balance, 0.1, 1.0)
        else:
            component_balance = 0.1

        total_accuracy = norm_accuracy * distance_accuracy * component_balance
        
        return total_accuracy

    @final
    def _normalize_angle(self, angle: float) -> float:
        """
        角度を-180°から180°の範囲に正規化
        
        Args:
            angle: 正規化する角度（度）
            
        Returns:
            -180°から180°の範囲に正規化された角度
        """
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle
    
    @final
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """
        2つの角度間の最小差分を計算（wrap-around考慮）
        
        Args:
            angle1, angle2: 比較する角度（度）
            
        Returns:
            -180°から180°の範囲での角度差分
        """
        diff = angle2 - angle1
        return self._normalize_angle(diff)
    
    @final
    def _smooth_yaw_angle(self, raw_yaw: float, confidence: float) -> float:
        """
        yaw角の平滑化処理（Priority 2実装）
        wrap-around問題を解決し、異常値を除去
        
        Args:
            raw_yaw: 生の推定yaw角（度）
            confidence: 推定の信頼度
            
        Returns:
            平滑化されたyaw角（度）
        """
        # 角度を-180°〜180°に正規化
        normalized_yaw = self._normalize_angle(raw_yaw)
        
        # 履歴が空の場合は初期値として設定
        if not self.yaw_history:
            self.yaw_history.append(normalized_yaw)
            return normalized_yaw
        
        # 前回の角度との差分を計算
        last_yaw = self.yaw_history[-1]
        angle_diff = abs(self._angle_difference(last_yaw, normalized_yaw))
        
        # 異常値検出：急激な変化（120°以上）は信頼度に応じて処理
        max_reasonable_change = 120.0
        if angle_diff > max_reasonable_change:
            # 信頼度が低い場合は前回値を継続
            if confidence < 0.8:
                return last_yaw
            
            # 信頼度が高い場合でも変化を制限
            if angle_diff > 150.0:
                change_direction = 1.0 if self._angle_difference(last_yaw, normalized_yaw) > 0 else -1.0
                limited_change = max_reasonable_change * change_direction
                normalized_yaw = self._normalize_angle(last_yaw + limited_change)
        
        # 移動平均による平滑化
        if len(self.yaw_history) >= 3:
            # 過去数点の重み付き平均
            weights = [0.1, 0.3, 0.6]  # 現在値により大きな重み
            recent_yaws = self.yaw_history[-2:] + [normalized_yaw]
            
            # wrap-aroundを考慮した重み付き平均
            smoothed_yaw = self._weighted_circular_mean(recent_yaws, weights)
        else:
            # 履歴が不十分な場合は単純平均
            all_yaws = self.yaw_history + [normalized_yaw]
            smoothed_yaw = self._circular_mean(all_yaws)
        
        # 履歴を更新
        self.yaw_history.append(smoothed_yaw)
        if len(self.yaw_history) > self.max_history_size:
            self.yaw_history.pop(0)
        
        return smoothed_yaw
    
    @final
    def _circular_mean(self, angles: list[float]) -> float:
        """
        円周角の平均を計算（wrap-around考慮）
        
        Args:
            angles: 角度のリスト（度）
            
        Returns:
            円周平均角度（度）
        """
        if not angles:
            return 0.0
        
        # 角度をラジアンに変換してベクトル化
        sin_sum = sum(np.sin(np.radians(angle)) for angle in angles)
        cos_sum = sum(np.cos(np.radians(angle)) for angle in angles)
        
        # 平均ベクトルから角度を計算
        mean_angle_rad = np.arctan2(sin_sum, cos_sum)
        return float(np.degrees(mean_angle_rad))
    
    @final
    def _weighted_circular_mean(self, angles: list[float], weights: list[float]) -> float:
        """
        重み付き円周角の平均を計算
        
        Args:
            angles: 角度のリスト（度）
            weights: 対応する重みのリスト
            
        Returns:
            重み付き円周平均角度（度）
        """
        if not angles or len(angles) != len(weights):
            return 0.0
        
        # 重み付きベクトル和
        sin_sum = sum(w * np.sin(np.radians(angle)) for angle, w in zip(angles, weights))
        cos_sum = sum(w * np.cos(np.radians(angle)) for angle, w in zip(angles, weights))
        
        # 平均ベクトルから角度を計算
        mean_angle_rad = np.arctan2(sin_sum, cos_sum)
        return float(np.degrees(mean_angle_rad))

    @final
    def _smooth_direction_vector(self, uwbp: UwbPData) -> UwbPData | None:
        """
        方向ベクトルの平滑化処理
        急激な変化を検出して外れ値を除去し、移動平均で平滑化
        """
        tag_id = uwbp["tag_id"]
        
        # 現在の方向ベクトル
        current_direction = np.array([
            uwbp["direction_vec_x"],
            uwbp["direction_vec_y"], 
            uwbp["direction_vec_z"]
        ])
        
        # 正規化
        current_norm = np.linalg.norm(current_direction)
        if current_norm <= 0:
            return None
        current_direction = current_direction / current_norm
        
        # タグIDの履歴を初期化
        if tag_id not in self.direction_history:
            self.direction_history[tag_id] = []
        
        history = self.direction_history[tag_id]
        
        # 外れ値検出：過去の方向ベクトルとの角度差をチェック
        is_outlier = False
        if len(history) > 0:
            # 直前の方向ベクトルとの角度差
            last_direction = history[-1]
            dot_product = np.clip(np.dot(current_direction, last_direction), -1.0, 1.0)
            angle_diff_rad = np.arccos(dot_product)
            angle_diff_deg = np.degrees(angle_diff_rad)
            
            # 30度以上の急激な変化は外れ値とみなす
            if angle_diff_deg > 30.0:
                is_outlier = True
        
        if not is_outlier:
            # 履歴に追加
            history.append(current_direction.copy())
            
            # 履歴サイズ制限
            if len(history) > self.max_direction_history_size:
                history.pop(0)
            
            # 移動平均で平滑化
            if len(history) >= 2:
                # ベクトルの平均（単純平均後に正規化）
                mean_direction = np.mean(history, axis=0)
                mean_norm = np.linalg.norm(mean_direction)
                if mean_norm > 0:
                    smoothed_direction = mean_direction / mean_norm
                else:
                    smoothed_direction = current_direction
            else:
                smoothed_direction = current_direction
        else:
            # 外れ値の場合は直前の値を使用
            if len(history) > 0:
                smoothed_direction = history[-1]
            else:
                smoothed_direction = current_direction
        
        # 平滑化されたUWBPデータを作成
        smoothed_uwbp = uwbp.copy()
        smoothed_uwbp["direction_vec_x"] = smoothed_direction[0]
        smoothed_uwbp["direction_vec_y"] = smoothed_direction[1] 
        smoothed_uwbp["direction_vec_z"] = smoothed_direction[2]
        
        return smoothed_uwbp

    @final
    def _calc_continuity_accuracy(self, uwbp: UwbPData, gpos: GposData) -> float:
        """
        連続性に基づく信頼度を計算
        過去の位置推定との連続性が高いほど信頼度を高くする
        """
        if len(self.previous_positions) == 0:
            return 1.0  # 初回は満点
        
        # 現在の推定位置を計算
        current_pos = self._uwb_to_global_pos_by_uwbp(gpos, uwbp)
        if current_pos is None:
            return 0.1  # 計算できない場合は低信頼度
        
        # 直前の位置との距離差
        last_pos = self.previous_positions[-1]
        distance_diff = np.linalg.norm(current_pos - last_pos)
        
        # 距離差による信頼度（移動速度が現実的かをチェック）
        # 1秒間に5m以上の移動は異常とみなす
        max_reasonable_move = 5.0  # m/s
        
        if distance_diff <= max_reasonable_move:
            continuity_score = 1.0
        elif distance_diff <= max_reasonable_move * 2:
            # 線形減少
            continuity_score = 1.0 - (distance_diff - max_reasonable_move) / max_reasonable_move
        else:
            continuity_score = 0.1  # 最低信頼度
        
        return float(max(0.1, float(continuity_score)))

    @final
    def _update_position_history(self, position: npt.NDArray[np.float64]) -> None:
        """
        位置履歴を更新
        """
        self.previous_positions.append(position.copy())
        
        # 履歴サイズ制限
        if len(self.previous_positions) > self.max_position_history_size:
            self.previous_positions.pop(0)

    @final
    def _remove_statistical_outliers(
        self, valid_data: list[tuple[npt.NDArray[np.float64], float]]
    ) -> list[tuple[npt.NDArray[np.float64], float]]:
        """
        統計的手法で外れ値を除去
        IQR（四分位範囲）を使用して外れ値を検出・除去
        """
        if len(valid_data) <= 2:
            return valid_data  # データ数が少ない場合はそのまま返す
        
        positions = np.array([d[0] for d in valid_data])
        
        # 各軸（x, y, z）について外れ値を検出
        outlier_indices = set()
        
        for axis in range(3):  # x, y, z軸
            axis_values = positions[:, axis]
            
            # 四分位数を計算
            q1 = np.percentile(axis_values, 25)
            q3 = np.percentile(axis_values, 75)
            iqr = q3 - q1
            
            # 外れ値の閾値
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 外れ値のインデックスを記録
            for i, val in enumerate(axis_values):
                if val < lower_bound or val > upper_bound:
                    outlier_indices.add(i)
        
        # 外れ値を除去した新しいデータリストを作成
        filtered_data = []
        for i, (pos, acc) in enumerate(valid_data):
            if i not in outlier_indices:
                filtered_data.append((pos, acc))
        
        # フィルタリング後にデータが1個以下になった場合は元のデータを返す
        if len(filtered_data) <= 1:
            return valid_data
        
        return filtered_data

    @final
    def _calibrate_device_orientation(self) -> None:
        """
        AHRSデータから重力方向を判定し、デバイス座標系の補正行列を作成
        """
        # AHRSデータを取得
        ahrs_data = self.ahrs_datarecorder.last_appended_data
        if not ahrs_data:
            # AHRSデータがない場合はデフォルト（補正なし）
            self.device_orientation_matrix = np.eye(3)
            self.gravity_calibrated = True
            return

        # 複数のAHRSデータから平均的な重力方向を計算
        gravity_vectors = []
        for ahrs in ahrs_data[-5:]:  # 最新5個のデータを使用
            quat = np.array([
                ahrs["quat_2"], ahrs["quat_3"], 
                ahrs["quat_4"], ahrs["quat_w"]
            ])
            quat = quat / np.linalg.norm(quat)
            
            # ワールド座標の重力方向（下向き = [0, 0, -1]）
            world_gravity = np.array([0, 0, -1])
            
            # デバイス座標での重力方向に変換
            rot = R.from_quat(quat)
            device_gravity = rot.inv().apply(world_gravity)
            gravity_vectors.append(device_gravity)

        # 平均重力方向を計算
        avg_gravity = np.mean(gravity_vectors, axis=0)
        avg_gravity = avg_gravity / np.linalg.norm(avg_gravity)

        # デバイス座標系補正行列を作成
        self.device_orientation_matrix = self._create_orientation_correction_matrix(avg_gravity)
        self.gravity_calibrated = True



    @final
    def _create_orientation_correction_matrix(self, gravity_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        重力方向から座標系補正行列を作成
        
        Args:
            gravity_vector: デバイス座標系での重力方向ベクトル
            
        Returns:
            3x3補正行列（デバイス座標→標準座標の変換）
        """
        # 支配軸を判定
        abs_gravity = np.abs(gravity_vector)
        dominant_axis = np.argmax(abs_gravity)
        
        # 標準座標系: X軸=前方, Y軸=上方, Z軸=右方
        # 重力方向に基づいて補正行列を作成
        
        if dominant_axis == 0:  # X軸が重力方向（縦向き）
            if gravity_vector[0] < 0:
                # 正常な縦向き: X軸=下方向
                # 補正: X→Y(上), Y→Z(右), Z→-X(前)
                correction_matrix = np.array([
                    [0, 0, -1],  # 新X(前) = -旧Z
                    [-1, 0, 0],  # 新Y(上) = -旧X  
                    [0, 1, 0]    # 新Z(右) = 旧Y
                ])
            else:
                # 逆向き縦向き: X軸=上方向
                # 補正: X→-Y(下), Y→-Z(左), Z→-X(前)
                correction_matrix = np.array([
                    [0, 0, -1],  # 新X(前) = -旧Z
                    [1, 0, 0],   # 新Y(上) = 旧X
                    [0, -1, 0]   # 新Z(右) = -旧Y
                ])
                
        elif dominant_axis == 1:  # Y軸が重力方向（平置き）
            if gravity_vector[1] < 0:
                # Y軸=下方向（正常）: 補正なし
                correction_matrix = np.eye(3)
            else:
                # Y軸=上方向（上下逆）: Y軸とZ軸を反転
                correction_matrix = np.array([
                    [1, 0, 0],   # 新X(前) = 旧X
                    [0, -1, 0],  # 新Y(上) = -旧Y
                    [0, 0, -1]   # 新Z(右) = -旧Z
                ])
                
        else:  # Z軸が重力方向（横向き）
            if gravity_vector[2] < 0:
                # Z軸=下方向: 横向き
                # 補正: X→Z(右), Y→X(前), Z→-Y(上)
                correction_matrix = np.array([
                    [0, 1, 0],   # 新X(前) = 旧Y
                    [0, 0, -1],  # 新Y(上) = -旧Z
                    [1, 0, 0]    # 新Z(右) = 旧X
                ])
            else:
                # Z軸=上方向: 逆横向き  
                correction_matrix = np.array([
                    [0, -1, 0],  # 新X(前) = -旧Y
                    [0, 0, 1],   # 新Y(上) = 旧Z
                    [-1, 0, 0]   # 新Z(右) = -旧X
                ])

        return correction_matrix

    @final
    def _apply_device_orientation_correction(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        デバイス座標系のベクトルを標準座標系に補正
        
        Args:
            vector: デバイス座標系のベクトル
            
        Returns:
            標準座標系に補正されたベクトル
        """
        if self.device_orientation_matrix is not None:
            return self.device_orientation_matrix @ vector
        else:
            return vector