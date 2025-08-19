from typing import final, Dict, List, Tuple, Optional, NamedTuple, Any
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
from bisect import bisect_left


from src.lib.recorder import DataRecorderProtocol
from src.type import Position

logger = logging.getLogger(__name__)


class TagEstimate:
    """各タグからの推定結果を格納するクラス"""

    def __init__(
        self,
        tag_id: str,
        position: np.ndarray,
        confidence: float,
        distance: float,
        method: str,
        is_los: bool = True,
    ):
        self.tag_id = tag_id
        self.position = position
        self.confidence = confidence
        self.distance = distance
        self.method = method  # 'UWBT' or 'UWBP'
        self.is_los = is_los  # Line of Sight フラグ


class PositionWithLOS(NamedTuple):
    """LOS/NLOS情報を含む位置データ"""

    x: float
    y: float
    z: float
    is_los: bool
    confidence: float
    method: str  # 'UWBT' or 'UWBP'
    gpos_distance: float = 0.0  # GPOSとの距離（メートル）
    is_far_from_gpos: bool = False  # GPOSから3m以上離れているかのフラグ


class UWBLocalizer(DataRecorderProtocol):
    """
    UWB による位置推定のためのクラス

    各タグごとに個別の推定軌跡を作成し、保持する。
    """

    def __init__(self, trial_id: str = "", logger: Optional[Any] = None) -> None:
        super().__init__()
        self.tag_trajectories: Dict[str, List[Position]] = (
            {}
        )  # タグごとの推定軌跡（互換性のため残す）
        self.tag_trajectories_with_los: Dict[str, List[PositionWithLOS]] = (
            {}
        )  # LOS情報付き軌跡
        self.tag_estimates: Dict[str, List[TagEstimate]] = {}  # タグごとの推定履歴
        self.nlos_threshold = 0.5  # NLOS判定の閾値
        self.max_history = 10  # 各タグの推定履歴の最大保持数
        self.current_tag_positions: Dict[str, Position] = {}  # 各タグの現在位置
        self.raw_measurements: Dict[str, List[PositionWithLOS]] = (
            {}
        )  # 生の測定値（重み付き平均なし）
        self.uwbt_only_measurements: Dict[str, List[PositionWithLOS]] = (
            {}
        )  # UWBTのみの測定値
        self.time_window_ms = 100.0  # 時間窓（ミリ秒）
        self.max_time_diff_ms = 500.0  # 最大許容時間差（ミリ秒）

    def interpolate_gpos_data(
        self, before: Any, after: Any, target_timestamp: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """2つのGPOSデータ間で補間を行う"""
        # タイムスタンプの比率を計算
        t_before = before["sensor_timestamp"]
        t_after = after["sensor_timestamp"]
        alpha = (target_timestamp - t_before) / (t_after - t_before)

        # 位置の線形補間
        loc_before = np.array(
            [before["location_x"], before["location_y"], before["location_z"]]
        )
        loc_after = np.array(
            [after["location_x"], after["location_y"], after["location_z"]]
        )
        location = loc_before + alpha * (loc_after - loc_before)

        # クォータニオンのSLERP（球面線形補間）- 簡易実装
        quat_before = np.array(
            [before["quat_x"], before["quat_y"], before["quat_z"], before["quat_w"]]
        )
        quat_after = np.array(
            [after["quat_x"], after["quat_y"], after["quat_z"], after["quat_w"]]
        )

        # 内積を計算
        dot = np.dot(quat_before, quat_after)

        # 負の場合は符号を反転
        if dot < 0:
            quat_after = -quat_after
            dot = -dot

        # ほぼ同じクォータニオンの場合は線形補間
        if dot > 0.9995:
            quat = quat_before + alpha * (quat_after - quat_before)
            quat = quat / np.linalg.norm(quat)
        else:
            # SLERPを計算
            theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
            theta = theta_0 * alpha
            quat_2 = quat_after - quat_before * dot
            quat_2 = quat_2 / np.linalg.norm(quat_2)
            quat = quat_before * np.cos(theta) + quat_2 * np.sin(theta)

        return location, quat

    def get_synchronized_tag_pose(
        self, tag_id: str, uwb_timestamp: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """UWBタイムスタンプに対応するタグ位置を取得（補間機能付き）"""
        gpos_data = self.gpos_datarecorder.data

        # タグIDでフィルタリング
        tag_gpos_data = [d for d in gpos_data if d["object_id"] == tag_id]

        if not tag_gpos_data:
            return None, None

        # タイムスタンプでソート（されていると仮定）
        timestamps = [d["sensor_timestamp"] for d in tag_gpos_data]

        # バイナリサーチでインデックスを探す
        idx = bisect_left(timestamps, uwb_timestamp)

        # 補間処理
        if idx == 0:
            # 最初のデータより前の場合
            closest_data = tag_gpos_data[0]
            time_diff_ms = abs(closest_data["sensor_timestamp"] - uwb_timestamp) * 1000
            if time_diff_ms > self.max_time_diff_ms:
                logger.debug(
                    f"GPOS data too old for tag {tag_id}: time_diff={time_diff_ms:.1f}ms"
                )
                return None, None
            # 補間せずに最初のデータを使用
            location = np.array(
                [
                    closest_data["location_x"],
                    closest_data["location_y"],
                    closest_data["location_z"],
                ]
            )
            quat = np.array(
                [
                    closest_data["quat_x"],
                    closest_data["quat_y"],
                    closest_data["quat_z"],
                    closest_data["quat_w"],
                ]
            )
            quat = quat / np.linalg.norm(quat)
            return location, quat

        elif idx == len(tag_gpos_data):
            # 最後のデータより後の場合
            closest_data = tag_gpos_data[-1]
            time_diff_ms = abs(closest_data["sensor_timestamp"] - uwb_timestamp) * 1000
            if time_diff_ms > self.max_time_diff_ms:
                logger.debug(
                    f"GPOS data too old for tag {tag_id}: time_diff={time_diff_ms:.1f}ms"
                )
                return None, None
            # 補間せずに最後のデータを使用
            location = np.array(
                [
                    closest_data["location_x"],
                    closest_data["location_y"],
                    closest_data["location_z"],
                ]
            )
            quat = np.array(
                [
                    closest_data["quat_x"],
                    closest_data["quat_y"],
                    closest_data["quat_z"],
                    closest_data["quat_w"],
                ]
            )
            quat = quat / np.linalg.norm(quat)
            return location, quat

        else:
            # 2つのデータの間にある場合
            before = tag_gpos_data[idx - 1]
            after = tag_gpos_data[idx]

            # 時間窓チェック
            time_diff_before = abs(before["sensor_timestamp"] - uwb_timestamp) * 1000
            time_diff_after = abs(after["sensor_timestamp"] - uwb_timestamp) * 1000

            # どちらかが時間窓内であれば補間を実行
            if (
                time_diff_before <= self.time_window_ms
                or time_diff_after <= self.time_window_ms
            ):
                # 補間を実行
                location, quat = self.interpolate_gpos_data(
                    before, after, uwb_timestamp
                )
                return location, quat
            elif time_diff_before <= self.max_time_diff_ms:
                # 前のデータを使用
                location = np.array(
                    [before["location_x"], before["location_y"], before["location_z"]]
                )
                quat = np.array(
                    [
                        before["quat_x"],
                        before["quat_y"],
                        before["quat_z"],
                        before["quat_w"],
                    ]
                )
                quat = quat / np.linalg.norm(quat)
                return location, quat
            elif time_diff_after <= self.max_time_diff_ms:
                # 後のデータを使用
                location = np.array(
                    [after["location_x"], after["location_y"], after["location_z"]]
                )
                quat = np.array(
                    [after["quat_x"], after["quat_y"], after["quat_z"], after["quat_w"]]
                )
                quat = quat / np.linalg.norm(quat)
                return location, quat
            else:
                logger.debug(
                    f"GPOS data too old for tag {tag_id}: min_time_diff={min(time_diff_before, time_diff_after):.1f}ms"
                )
                return None, None

    def estimate_from_uwbt(self, tag_id: str) -> List[TagEstimate]:
        """UWBTデータから指定タグの位置を推定"""
        estimates = []
        uwbt_data = self.uwbt_datarecorder.data

        # 最新のデータから処理（最大10個）
        for data in uwbt_data[-10:]:
            if data["tag_id"] != tag_id:
                continue

            # NLOS判定（1.0がNLOS、0.0がLOS）
            nlos_value = data.get("nlos", 0.0)
            is_los = nlos_value < self.nlos_threshold

            # NLOSでもデータを使用するが、信頼度を下げる
            if not is_los:
                logger.debug(
                    f"UWBT - Tag {tag_id}: NLOS detected (value={nlos_value:.2f})"
                )

            # タイムスタンプベースでタグ位置を取得
            uwb_timestamp = data["sensor_timestamp"]
            tag_loc, tag_quat = self.get_synchronized_tag_pose(tag_id, uwb_timestamp)
            if tag_loc is None or tag_quat is None:
                continue

            # 球面座標からローカルデカルト座標への変換
            distance = data["distance"]
            azimuth_rad = np.radians(data["aoa_azimuth"])
            elevation_rad = np.radians(data["aoa_elevation"])

            # ローカル座標系での相対位置
            x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            y = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            z = distance * np.sin(elevation_rad)
            local_point = np.array([x, y, z])

            # タグの姿勢で回転してワールド座標に変換
            R_tag = R.from_quat(tag_quat)
            world_point = R_tag.apply(local_point) + tag_loc

            # 信頼度計算（距離が遠いほど信頼度低下）
            confidence = 1.0 / (1.0 + distance * 0.1)
            # NLOSの場合は信頼度を下げる（1.0の場合は80%、0.0の場合は100%）
            if is_los:
                confidence *= 1.0  # LOSの場合は信頼度を維持
            else:
                confidence *= 0.8  # NLOSの場合は信頼度を20%低減（以前は50%だった）

            estimate = TagEstimate(
                tag_id=tag_id,
                position=world_point,
                confidence=confidence,
                distance=distance,
                method="UWBT",
                is_los=is_los,
            )
            estimates.append(estimate)

            logger.debug(
                f"UWBT - Tag {tag_id}: pos=({world_point[0]:.3f}, {world_point[1]:.3f}, {world_point[2]:.3f}), "
                f"conf={confidence:.3f}, dist={distance:.3f}m"
            )

        return estimates

    def estimate_from_uwbp(self, tag_id: str) -> List[TagEstimate]:
        """UWBPデータから指定タグの位置を推定"""
        estimates = []
        uwbp_data = self.uwbp_datarecorder.data

        # 最新のデータから処理（最大10個）
        for data in uwbp_data[-10:]:
            if data["tag_id"] != tag_id:
                continue

            # タイムスタンプベースでタグ位置を取得
            uwb_timestamp = data["sensor_timestamp"]
            tag_loc, tag_quat = self.get_synchronized_tag_pose(tag_id, uwb_timestamp)
            if tag_loc is None or tag_quat is None:
                continue

            # 方向ベクトルと距離から位置を計算
            distance = data["distance"]
            direction_vec = np.array(
                [
                    data["direction_vec_x"],
                    data["direction_vec_y"],
                    data["direction_vec_z"],
                ]
            )

            # 方向ベクトルを正規化
            direction_norm = np.linalg.norm(direction_vec)
            if direction_norm > 0:
                direction_vec = direction_vec / direction_norm
            else:
                continue

            # ローカル座標での相対位置
            local_point = distance * direction_vec

            # タグの姿勢で回転してワールド座標に変換
            R_tag = R.from_quat(tag_quat)
            world_point = R_tag.apply(local_point) + tag_loc

            # 信頼度計算
            confidence = 1.0 / (1.0 + distance * 0.1)

            estimate = TagEstimate(
                tag_id=tag_id,
                position=world_point,
                confidence=confidence,
                distance=distance,
                method="UWBP",
                is_los=True,  # UWBPはNLOS情報なし
            )
            estimates.append(estimate)

            logger.debug(
                f"UWBP - Tag {tag_id}: pos=({world_point[0]:.3f}, {world_point[1]:.3f}, {world_point[2]:.3f}), "
                f"conf={confidence:.3f}, dist={distance:.3f}m"
            )

        return estimates

    def estimate_position_per_tag(
        self, tag_id: str
    ) -> Optional[Tuple[Position, float, int]]:
        """指定されたタグからのデータのみを使用した位置推定"""
        # UWBTとUWBPの両方から推定を取得
        uwbt_estimates = self.estimate_from_uwbt(tag_id)
        uwbp_estimates = self.estimate_from_uwbp(tag_id)

        all_estimates = uwbt_estimates + uwbp_estimates

        if not all_estimates:
            return None

        # 重み付き平均で位置を計算
        positions = np.array([est.position for est in all_estimates])
        weights = np.array([est.confidence for est in all_estimates])

        # UWBTの重みを1.2倍（より精度が高いため）
        for i, est in enumerate(all_estimates):
            if est.method == "UWBT":
                weights[i] *= 1.2

        total_weight = np.sum(weights)
        if total_weight > 0:
            weighted_position = (
                np.sum(positions * weights.reshape(-1, 1), axis=0) / total_weight
            )

            # Positionオブジェクトに変換
            final_position = Position(
                x=float(weighted_position[0]),
                y=float(weighted_position[1]),
                z=float(weighted_position[2]),
            )

            # タグの推定履歴を更新
            if tag_id not in self.tag_estimates:
                self.tag_estimates[tag_id] = []

            # 最も信頼度の高い推定を履歴に追加
            best_estimate = max(all_estimates, key=lambda e: e.confidence)
            self.tag_estimates[tag_id].append(best_estimate)

            # 履歴の最大数を維持
            if len(self.tag_estimates[tag_id]) > self.max_history:
                self.tag_estimates[tag_id] = self.tag_estimates[tag_id][
                    -self.max_history :
                ]

            return final_position, float(total_weight), len(all_estimates)

        return None

    def estimate_uwb_blue_only(self) -> Position | None:
        """青色の点のみを使用したUWB位置推定（通常のestimate_uwbを実行後に青色フィルタリング）"""
        try:
            # まず通常のUWB推定を実行してデータを蓄積
            regular_position = self.estimate_uwb()

            # 青色の点が十分にある場合は青色のみの推定を試みる
            blue_position = self.get_best_blue_estimate()

            if blue_position is not None:
                logger.info(
                    f"青色のみUWB推定結果: ({blue_position.x:.3f}, {blue_position.y:.3f}, {blue_position.z:.3f})"
                )
                return blue_position
            else:
                # 青色の点が不足している場合は通常の推定結果を使用
                logger.debug("青色の点が不足 - 通常の推定結果を使用")
                return regular_position

        except Exception as e:
            logger.error(f"青色のみUWB推定でエラー: {e}")
            # エラー時は前回の位置を返す
            if self.current_tag_positions:
                return list(self.current_tag_positions.values())[0]
            return Position(0.0, 0.0, 0.0)

    @final
    def estimate_uwb(self) -> Position | None:
        """各タグごとに個別の推定軌跡を作成し、最も信頼度の高いタグの位置を返す"""
        try:
            # 利用可能な全タグIDを収集
            all_tag_ids = set()

            # UWBTデータからタグIDを収集
            for uwbt_data in self.uwbt_datarecorder.data[-20:]:
                all_tag_ids.add(uwbt_data["tag_id"])

            # UWBPデータからタグIDを収集
            for uwbp_data in self.uwbp_datarecorder.data[-20:]:
                all_tag_ids.add(uwbp_data["tag_id"])

            if not all_tag_ids:
                logger.debug("No UWB data available")
                # 前回の位置を返す（最も信頼度の高いタグの位置）
                if self.current_tag_positions:
                    return list(self.current_tag_positions.values())[0]
                return Position(0.0, 0.0, 0.0)

            # 各タグごとに推定を行い、軌跡を更新
            best_tag_id = None
            best_confidence = 0.0
            best_position = None

            for tag_id in all_tag_ids:
                # 生の測定値を保存（重み付き平均前）
                uwbt_estimates = self.estimate_from_uwbt(tag_id)
                uwbp_estimates = self.estimate_from_uwbp(tag_id)
                all_raw_estimates = uwbt_estimates + uwbp_estimates

                # 生の測定値を軌跡として保存
                if tag_id not in self.raw_measurements:
                    self.raw_measurements[tag_id] = []

                # UWBTのみの測定値を保存
                if tag_id not in self.uwbt_only_measurements:
                    self.uwbt_only_measurements[tag_id] = []

                for estimate in all_raw_estimates:
                    # 各推定点でGPOSとの距離を計算
                    gpos_distance = 0.0
                    is_far_from_gpos = False

                    # 最新のGPOSデータを取得
                    gpos_data = self.gpos_datarecorder.data
                    tag_gpos_data = [d for d in gpos_data if d["object_id"] == tag_id]

                    if tag_gpos_data:
                        # 最新のGPOSデータを使用
                        latest_gpos = tag_gpos_data[-1]
                        gpos_pos = np.array(
                            [
                                latest_gpos["location_x"],
                                latest_gpos["location_y"],
                                latest_gpos["location_z"],
                            ]
                        )
                        uwb_pos = np.array(
                            [
                                estimate.position[0],
                                estimate.position[1],
                                estimate.position[2],
                            ]
                        )

                        # ユークリッド距離を計算
                        gpos_distance = float(np.linalg.norm(uwb_pos - gpos_pos))
                        is_far_from_gpos = gpos_distance >= 3.0

                    raw_position = PositionWithLOS(
                        x=float(estimate.position[0]),
                        y=float(estimate.position[1]),
                        z=float(estimate.position[2]),
                        is_los=estimate.is_los,
                        confidence=estimate.confidence,
                        method=estimate.method,
                        gpos_distance=gpos_distance,
                        is_far_from_gpos=is_far_from_gpos,
                    )
                    self.raw_measurements[tag_id].append(raw_position)

                    # UWBTのみの場合は別途保存
                    if estimate.method == "UWBT":
                        self.uwbt_only_measurements[tag_id].append(raw_position)

                # 既存の重み付き平均処理
                result = self.estimate_position_per_tag(tag_id)
                if result is not None:
                    position, confidence, count = result

                    # タグの軌跡を更新（互換性のため両方更新）
                    if tag_id not in self.tag_trajectories:
                        self.tag_trajectories[tag_id] = []
                    self.tag_trajectories[tag_id].append(position)

                    # LOS情報付き軌跡を更新
                    if tag_id not in self.tag_trajectories_with_los:
                        self.tag_trajectories_with_los[tag_id] = []

                    # GPOSとの距離を計算
                    gpos_distance = 0.0
                    is_far_from_gpos = False

                    # 最新のGPOSデータを取得
                    gpos_data = self.gpos_datarecorder.data
                    tag_gpos_data = [d for d in gpos_data if d["object_id"] == tag_id]

                    if tag_gpos_data:
                        # 最新のGPOSデータを使用
                        latest_gpos = tag_gpos_data[-1]
                        gpos_pos = np.array(
                            [
                                latest_gpos["location_x"],
                                latest_gpos["location_y"],
                                latest_gpos["location_z"],
                            ]
                        )
                        uwb_pos = np.array([position.x, position.y, position.z])

                        # ユークリッド距離を計算
                        gpos_distance = float(np.linalg.norm(uwb_pos - gpos_pos))
                        is_far_from_gpos = gpos_distance >= 3.0

                        if is_far_from_gpos:
                            logger.warning(
                                f"Tag {tag_id}: UWB position is {gpos_distance:.2f}m away from GPOS (threshold: 3.0m)"
                            )

                    # 最新の推定結果からLOS情報を取得
                    if tag_id in self.tag_estimates and self.tag_estimates[tag_id]:
                        latest_estimate = self.tag_estimates[tag_id][-1]
                        position_with_los = PositionWithLOS(
                            x=position.x,
                            y=position.y,
                            z=position.z,
                            is_los=latest_estimate.is_los,
                            confidence=confidence,
                            method=latest_estimate.method,
                            gpos_distance=gpos_distance,
                            is_far_from_gpos=is_far_from_gpos,
                        )
                    else:
                        # デフォルト値を使用
                        position_with_los = PositionWithLOS(
                            x=position.x,
                            y=position.y,
                            z=position.z,
                            is_los=True,
                            confidence=confidence,
                            method="UNKNOWN",
                            gpos_distance=gpos_distance,
                            is_far_from_gpos=is_far_from_gpos,
                        )

                    self.tag_trajectories_with_los[tag_id].append(position_with_los)

                    # 現在位置を更新
                    self.current_tag_positions[tag_id] = position

                    # 最も信頼度の高いタグを記録
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_position = position
                        best_tag_id = tag_id

                    logger.info(
                        f"Tag {tag_id}: pos=({position.x:.3f}, {position.y:.3f}, {position.z:.3f}), "
                        f"conf={confidence:.3f}, count={count}, trajectory_points={len(self.tag_trajectories[tag_id])}"
                    )

            if best_position is None:
                logger.debug("No valid tag estimates available")
                if self.current_tag_positions:
                    return list(self.current_tag_positions.values())[0]
                return Position(0.0, 0.0, 0.0)

            logger.info(f"=== UWB Position Estimation Results ===")
            logger.info(f"Active tags: {len(all_tag_ids)}")
            logger.info(
                f"Best tag: {best_tag_id} with confidence {best_confidence:.3f}"
            )
            logger.info(
                f"Position: ({best_position.x:.3f}, {best_position.y:.3f}, {best_position.z:.3f})"
            )
            logger.info(f"=========================================")

            return best_position

        except Exception as e:
            logger.error(f"Error in UWB estimation: {e}")
            if self.current_tag_positions:
                return list(self.current_tag_positions.values())[0]
            return Position(0.0, 0.0, 0.0)

    def estimate_position_from_blue_points(self, tag_id: str) -> Optional[Position]:
        """青色の点のみから位置を推定（重み付き平均）"""
        blue_points = self.get_blue_only_measurements_for_tag(tag_id)

        if not blue_points:
            logger.debug(f"Tag {tag_id}: 青色の点がありません")
            return None

        # 信頼度による重み付き平均を計算
        positions = np.array([[pos.x, pos.y, pos.z] for pos in blue_points])
        weights = np.array([pos.confidence for pos in blue_points])

        # 重み正規化
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight

            # 重み付き平均
            weighted_position = np.sum(positions * weights.reshape(-1, 1), axis=0)

            estimated_position = Position(
                x=float(weighted_position[0]),
                y=float(weighted_position[1]),
                z=float(weighted_position[2]),
            )

            logger.debug(
                f"Tag {tag_id}: 青色点からの推定位置 "
                f"({estimated_position.x:.3f}, {estimated_position.y:.3f}, {estimated_position.z:.3f}) "
                f"from {len(blue_points)} blue points"
            )

            return estimated_position

        return None

    def get_tag_trajectories(self) -> Dict[str, List[Position]]:
        """全タグの推定軌跡を取得"""
        return self.tag_trajectories.copy()

    def get_tag_trajectories_with_los(self) -> Dict[str, List[PositionWithLOS]]:
        """全タグのLOS情報付き推定軌跡を取得"""
        return self.tag_trajectories_with_los.copy()

    def get_raw_measurements(self) -> Dict[str, List[PositionWithLOS]]:
        """生の測定値（重み付き平均なし）を取得"""
        return self.raw_measurements.copy()

    def get_blue_only_measurements_for_tag(self, tag_id: str) -> List[PositionWithLOS]:
        """指定されたタグの青色の点のみを取得"""
        measurements = self.raw_measurements.get(tag_id, [])
        return [pos for pos in measurements if pos.is_los and not pos.is_far_from_gpos]

    def create_blue_only_trajectories(self) -> Dict[str, List[Position]]:
        """青色の点のみから推定軌跡を作成（重み付き平均後の軌跡から青色フィルタリング）"""
        blue_trajectories = {}

        # 重み付き平均後のLOS情報付き軌跡から青色の点をフィルタリング
        tag_trajectories_with_los = self.get_tag_trajectories_with_los()

        for tag_id, trajectory in tag_trajectories_with_los.items():
            if not trajectory:
                continue

            # 青色の条件でフィルタリング（LOS & GPOSから3m以内）
            blue_points = [
                pos for pos in trajectory if pos.is_los and not pos.is_far_from_gpos
            ]

            if not blue_points:
                logger.info(f"Tag {tag_id}: 青色条件を満たす軌跡点がありません")
                continue

            # Position形式に変換
            blue_trajectory = [
                Position(x=pos.x, y=pos.y, z=pos.z) for pos in blue_points
            ]

            blue_trajectories[tag_id] = blue_trajectory

            logger.info(
                f"青色軌跡を作成 - Tag {tag_id}: {len(blue_trajectory)} points "
                f"(全軌跡 {len(trajectory)} → フィルタリング後 {len(blue_trajectory)}, "
                f"{len(blue_trajectory)/len(trajectory)*100:.1f}%)"
            )

        return blue_trajectories

    def get_best_blue_estimate(self) -> Optional[Position]:
        """青色軌跡から最新の最も信頼度の高い推定位置を取得"""
        best_position = None
        best_confidence = 0.0
        best_tag_id = None

        # 重み付き平均後の青色軌跡を取得
        tag_trajectories_with_los = self.get_tag_trajectories_with_los()

        for tag_id, trajectory in tag_trajectories_with_los.items():
            if not trajectory:
                continue

            # 青色の点のみを抽出
            blue_points = [
                pos for pos in trajectory if pos.is_los and not pos.is_far_from_gpos
            ]

            if not blue_points:
                continue

            # 最新の青色の点を使用
            latest_blue_point = blue_points[-1]

            # 青色の点の平均信頼度を計算
            avg_confidence = sum(pos.confidence for pos in blue_points) / len(
                blue_points
            )

            # 点数によるボーナス（多くの青色点がある方が信頼度が高い）
            point_bonus = min(len(blue_points) * 0.1, 0.5)
            total_confidence = avg_confidence + point_bonus

            if total_confidence > best_confidence:
                best_confidence = total_confidence
                best_position = Position(
                    x=latest_blue_point.x, y=latest_blue_point.y, z=latest_blue_point.z
                )
                best_tag_id = tag_id

        if best_position:
            logger.info(
                f"青色軌跡からの最良推定: Tag {best_tag_id}, "
                f"信頼度 {best_confidence:.3f}, "
                f"位置 ({best_position.x:.3f}, {best_position.y:.3f}, {best_position.z:.3f})"
            )

        return best_position

    def save_blue_only_trajectories_to_csv(
        self, output_dir: str, trial_id: str, timestamp: str
    ) -> None:
        """青色の点のみから作成した軌跡をCSVファイルに保存"""
        try:
            import pandas as pd
            from pathlib import Path

            output_path = Path(output_dir)
            blue_trajectories = self.create_blue_only_trajectories()

            for tag_id, trajectory in blue_trajectories.items():
                if not trajectory:
                    continue

                # DataFrame作成
                blue_df = pd.DataFrame(
                    [(pos.x, pos.y, pos.z) for pos in trajectory],
                    columns=["x", "y", "z"],
                )

                # CSVファイルに保存
                csv_file = (
                    output_path / f"{trial_id}_{timestamp}_tag_{tag_id}_blue_only.csv"
                )
                blue_df.to_csv(csv_file, index=False)

                logger.info(
                    f"青色のみ軌跡をCSV保存: {csv_file} ({len(trajectory)} points)"
                )

                # 統計情報をログ出力
                total_measurements = len(self.raw_measurements.get(tag_id, []))
                filter_ratio = (
                    (len(trajectory) / total_measurements * 100)
                    if total_measurements > 0
                    else 0
                )

                logger.info(
                    f"Tag {tag_id} フィルタリング統計: "
                    f"{total_measurements} → {len(trajectory)} points ({filter_ratio:.1f}%)"
                )

        except Exception as e:
            logger.error(f"青色のみ軌跡のCSV保存中にエラー: {e}")
            import traceback

            traceback.print_exc()

    def plot_blue_only_trajectories(
        self, output_dir: str = "output", map_file: str = "map/miraikan_5.bmp"
    ) -> None:
        """青色（LOS & GPOSから3m以内）の軌跡のみを表示する独立した可視化機能"""
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            import numpy as np
            from pathlib import Path
            import time

            # 出力ディレクトリの作成
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # タイムスタンプ付きのファイル名
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # マップ画像を読み込み
            src_dir = Path().resolve()
            bitmap_array = np.array(Image.open(src_dir / map_file)) / 255.0

            # マップの設定
            map_origin = (-5.625, -12.75)
            map_ppm = 100
            height, width = bitmap_array.shape[:2]
            width_m = width / map_ppm
            height_m = height / map_ppm

            extent = (
                map_origin[0],
                map_origin[0] + width_m,
                map_origin[1],
                map_origin[1] + height_m,
            )

            # 生の測定値を取得
            raw_measurements = self.get_raw_measurements()

            if not raw_measurements:
                logger.info("青色のみ表示: 測定データがありません")
                return

            # 各タグごとに青色のみの軌跡をプロット
            for tag_id, measurements in raw_measurements.items():
                if not measurements:
                    continue

                # 青色の条件でフィルタリング（LOS & GPOSから3m以内）
                blue_points = [
                    pos
                    for pos in measurements
                    if pos.is_los and not pos.is_far_from_gpos
                ]

                if not blue_points:
                    logger.info(f"Tag {tag_id}: 青色の条件を満たす点がありません")
                    continue

                # 図を作成
                fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                ax.imshow(bitmap_array, extent=extent, alpha=0.5, cmap="gray")

                # 青色の点のみをプロット
                x_coords = [pos.x for pos in blue_points]
                y_coords = [pos.y for pos in blue_points]

                # 軌跡の線をプロット（グレー）
                ax.plot(
                    x_coords,
                    y_coords,
                    "-",
                    color="gray",
                    alpha=0.3,
                    linewidth=1,
                    label=f"Tag {tag_id} trajectory",
                )

                # 青色の点をプロット
                ax.scatter(
                    x_coords,
                    y_coords,
                    s=20,
                    color="blue",
                    alpha=0.7,
                    edgecolors="darkblue",
                    linewidth=0.5,
                    label=f"Reliable Points: LOS & <3m from GPOS ({len(blue_points)} points)",
                    zorder=3,
                )

                # 始点と終点を強調
                if blue_points:
                    # 始点（緑の四角）
                    ax.scatter(
                        blue_points[0].x,
                        blue_points[0].y,
                        s=200,
                        color="green",
                        marker="s",
                        edgecolors="black",
                        linewidth=2,
                        label="Start",
                        zorder=5,
                    )

                    # 終点（オレンジの三角）
                    ax.scatter(
                        blue_points[-1].x,
                        blue_points[-1].y,
                        s=200,
                        color="orange",
                        marker="^",
                        edgecolors="black",
                        linewidth=2,
                        label="End",
                        zorder=5,
                    )

                # 軸とタイトルの設定
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_title(
                    f"Tag {tag_id} - Blue Only (Reliable Points)\n"
                    f"Total measurements: {len(measurements)} → Filtered: {len(blue_points)} "
                    f"({len(blue_points)/len(measurements)*100:.1f}%)"
                )
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right")

                # 統計情報をテキストボックスで表示
                info_text = f"Blue points: {len(blue_points)}\n"
                info_text += f"Total measurements: {len(measurements)}\n"
                info_text += (
                    f"Filter ratio: {len(blue_points)/len(measurements)*100:.1f}%\n"
                )
                info_text += f"All LOS & <3m from GPOS"

                ax.text(
                    0.02,
                    0.98,
                    info_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                )

                # ファイルに保存
                output_file = output_path / f"blue_only_tag_{tag_id}_{timestamp}.png"
                plt.savefig(output_file, bbox_inches="tight", dpi=150)
                plt.close(fig)

                logger.info(
                    f"青色のみ軌跡を保存: {output_file} ({len(blue_points)} points)"
                )

            logger.info(f"青色のみ表示が完了しました（出力: {output_path}）")

        except Exception as e:
            logger.error(f"青色のみ表示中にエラーが発生: {e}")
            import traceback

            traceback.print_exc()
