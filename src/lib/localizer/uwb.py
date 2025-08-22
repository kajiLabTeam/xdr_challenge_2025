from typing import final, Optional, NamedTuple, Any
import numpy as np
from scipy.spatial.transform import Rotation as R
from bisect import bisect_left
from src.lib.params._params import Params
from src.lib.recorder import DataRecorderProtocol
from src.type import Position
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time


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


class PositionWithLOS(NamedTuple):
    """LOS/NLOS情報を含む位置データ"""

    x: float
    y: float
    z: float
    is_nlos: bool
    confidence: float
    method: str  # 'UWBT' or 'UWBP'
    gpos_distance: float = 0.0  # GPOSとの距離（メートル）
    is_far_from_gpos: bool = False  # GPOSから3m以上離れているかのフラグ


class UWBLocalizer(DataRecorderProtocol):
    """
    UWB による位置推定のためのクラス

    各タグごとに個別の推定軌跡を作成し、保持する。
    """

    tag_trajectories: dict[str, list[Position]] = (
        {}
    )  # タグごとの推定軌跡（互換性のため残す）
    tag_trajectories_with_los: dict[str, list[PositionWithLOS]] = {}  # LOS情報付き軌跡
    tag_estimates: dict[str, list[TagEstimate]] = {}  # タグごとの推定履歴
    max_history = 10  # 各タグの推定履歴の最大保持数
    current_tag_positions: dict[str, Position] = {}  # 各タグの現在位置
    raw_measurements: dict[str, list[PositionWithLOS]] = (
        {}
    )  # 生の測定値（重み付き平均なし）
    uwbt_only_measurements: dict[str, list[PositionWithLOS]] = {}  # UWBTのみの測定値
    time_window_ms = 100.0  # 時間窓（ミリ秒）
    max_time_diff_ms = 500.0  # 最大許容時間差（ミリ秒）

    @final
    def estimate_uwb(self) -> Position | None:
        """青色の点のみを使用したUWB位置推定（通常のestimate_uwbを実行後に青色フィルタリング）"""
        try:
            # まず通常のUWB推定を実行してデータを蓄積
            self._uwb_generate_trajectories()

            # 青色の点が十分にある場合は青色のみの推定を試みる
            blue_position = self._uwb_get_best_blue_estimate()

            if blue_position is not None:
                return blue_position
            else:
                return None
        except Exception:
            return None

    @final
    def get_tag_trajectories(self) -> dict[str, list[Position]]:
        """全タグの推定軌跡を取得"""
        return self.tag_trajectories.copy()

    @final
    def plot_blue_only_trajectories(
        self, output_dir: str = "output", map_file: str = "map/miraikan_5.bmp"
    ) -> None:
        """青色（LOS & GPOSから3m以内）の軌跡のみを表示する独立した可視化機能"""
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
        raw_measurements = self.raw_measurements.copy()

        if not raw_measurements:
            return

        # 各タグごとに青色のみの軌跡をプロット
        for tag_id, measurements in raw_measurements.items():
            if not measurements:
                continue

            # 青色の条件でフィルタリング（LOS & GPOSから3m以内）
            blue_points = [
                pos
                for pos in measurements
                if not pos.is_nlos and not pos.is_far_from_gpos
            ]

            if not blue_points:
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

    @final
    def _uwb_estimate_from_uwbt(self) -> list[TagEstimate]:
        """UWBTデータから指定タグの位置を推定"""
        estimates: list[TagEstimate] = []
        uwbt_data = self.uwbt_datarecorder.last_appended_data

        # 最新のデータから処理（最大10個）
        for data in uwbt_data:
            # NLOS判定（1.0がNLOS、0.0がLOS）
            is_nlos = data["nlos"]

            # タイムスタンプベースでタグ位置を取得
            uwb_timestamp = data["sensor_timestamp"]
            tag_loc, tag_quat = self._uwb_get_synchronized_tag_pose(
                data["tag_id"], uwb_timestamp
            )
            if tag_loc is None or tag_quat is None:
                continue

            # 球面座標からローカルデカルト座標への変換
            distance = data["distance"]
            azimuth_rad = np.deg2rad(data["aoa_azimuth"])
            elevation_rad = np.deg2rad(data["aoa_elevation"])

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
            if is_nlos:
                confidence *= 0.8  # NLOSの場合は信頼度を20%低減（以前は50%だった）
            else:
                confidence *= 1.0  # LOSの場合は信頼度を維持

            estimate = TagEstimate(
                tag_id=data["tag_id"],
                position=world_point,
                confidence=confidence,
                distance=distance,
                method="UWBT",
                is_nlos=is_nlos,
            )
            estimates.append(estimate)

        return estimates

    @final
    def _uwb_interpolate_gpos_data(
        self, before: Any, after: Any, target_timestamp: float
    ) -> tuple[np.ndarray, np.ndarray]:
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

    @final
    def _uwb_get_synchronized_tag_pose(
        self, tag_id: str, uwb_timestamp: float
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
                location, quat = self._uwb_interpolate_gpos_data(
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
                return None, None

    @final
    def _uwb_estimate_from_uwbp(self) -> list[TagEstimate]:
        """UWBPデータから指定タグの位置を推定"""
        estimates = []
        uwbp_data = self.uwbp_datarecorder.last_appended_data

        # 最新のデータから処理（最大10個）
        for data in uwbp_data:
            # タイムスタンプベースでタグ位置を取得
            uwb_timestamp = data["sensor_timestamp"]
            tag_loc, tag_quat = self._uwb_get_synchronized_tag_pose(
                data["tag_id"], uwb_timestamp
            )
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
                tag_id=data["tag_id"],
                position=world_point,
                confidence=confidence,
                distance=distance,
                method="UWBP",
                is_nlos=False,  # UWBPはNLOS情報なし
            )
            estimates.append(estimate)

        return estimates

    @final
    def _uwb_estimate_position_per_tag(
        self, tag_id: str
    ) -> Optional[tuple[Position, float]]:
        """指定されたタグからのデータのみを使用した位置推定"""

        # UWBTとUWBPの両方から推定を取得
        uwbt_estimates = self.__uwb_estimate_from_uwbt()
        uwbp_estimates = self.__uwb_estimate_from_uwbp()

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

            # 最も信頼度の高い推定を履歴に追加
            best_estimate = max(all_estimates, key=lambda e: e.confidence)
            self.tag_estimates[tag_id].append(best_estimate)

            # 履歴の最大数を維持
            if len(self.tag_estimates[tag_id]) > self.max_history:
                self.tag_estimates[tag_id] = self.tag_estimates[tag_id][
                    -self.max_history :
                ]

            return final_position, float(total_weight)

        return None

    @final
    def _uwb_generate_trajectories(self) -> Position | None:
        """
        各タグごとに個別の推定軌跡を作成し、最も信頼度の高いタグの位置を返す
        TODO: 返り値は使用しないため削除する
        """
        try:
            # 利用可能な全タグIDを収集
            tag_ids = set(
                self.uwbp_datarecorder.tag_ids | self.uwbt_datarecorder.tag_ids
            )

            # 各タグごとに推定を行い、軌跡を更新
            best_confidence = 0.0
            best_position = None

            # 初期化
            for tag_id in tag_ids:
                if tag_id not in self.uwbt_only_measurements:
                    self.uwbt_only_measurements[tag_id] = []

                if tag_id not in self.raw_measurements:
                    self.raw_measurements[tag_id] = []

                if tag_id not in self.tag_estimates:
                    self.tag_estimates[tag_id] = []

            for tag_id in tag_ids:  # TODO
                # 生の測定値を保存（重み付き平均前）
                uwbt_estimated_positions = self._uwb_estimate_from_uwbt()
                uwbp_estimated_positions = self._uwb_estimate_from_uwbp()
                estimated_positions = (
                    uwbt_estimated_positions + uwbp_estimated_positions
                )

                for estimated_position in estimated_positions:
                    # ユークリッド距離を計算
                    is_far_from_gpos = (
                        estimated_position.distance >= Params.uwb_far_distance()
                    )

                    raw_position = PositionWithLOS(
                        x=float(estimated_position.position[0]),
                        y=float(estimated_position.position[1]),
                        z=float(estimated_position.position[2]),
                        is_nlos=estimated_position.is_nlos,
                        confidence=estimated_position.confidence,
                        method=estimated_position.method,
                        gpos_distance=estimated_position.distance,
                        is_far_from_gpos=is_far_from_gpos,
                    )
                    self.raw_measurements[tag_id].append(raw_position)

                    # UWBTのみの場合は別途保存
                    if estimated_position.method == "UWBT":
                        self.uwbt_only_measurements[tag_id].append(raw_position)

                # 既存の重み付き平均処理
                result = self._uwb_estimate_position_per_tag(
                    tag_id
                )  # TODO: tag_id を消した
                if result is not None:
                    position, confidence = result

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

                    # 最新の推定結果からLOS情報を取得
                    if tag_id in self.tag_estimates and self.tag_estimates[tag_id]:
                        latest_estimate = self.tag_estimates[tag_id][-1]
                        position_with_los = PositionWithLOS(
                            x=position.x,
                            y=position.y,
                            z=position.z,
                            is_nlos=latest_estimate.is_nlos,
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
                            is_nlos=False,
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

            if best_position is None:
                if self.current_tag_positions:
                    return list(self.current_tag_positions.values())[0]
                return Position(0.0, 0.0, 0.0)

            return best_position

        except Exception as e:
            if self.current_tag_positions:
                return list(self.current_tag_positions.values())[0]
            return Position(0.0, 0.0, 0.0)

    @final
    def _uwb_get_best_blue_estimate(self) -> Optional[Position]:
        """青色軌跡から最新の最も信頼度の高い推定位置を取得"""
        best_position = None
        best_confidence = 0.0
        best_tag_id = None

        # 重み付き平均後の青色軌跡を取得
        tag_trajectories_with_los = self.tag_trajectories_with_los.copy()

        for tag_id, trajectory in tag_trajectories_with_los.items():
            if not trajectory:
                continue

            # 青色の点のみを抽出
            blue_points = [
                pos
                for pos in trajectory
                if not pos.is_nlos and not pos.is_far_from_gpos
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

        return best_position

    @final
    def __uwb_estimate_from_uwbt(self) -> list[TagEstimate]:
        """UWBTデータから指定タグの位置を推定"""
        estimates: list[TagEstimate] = []
        uwbt_data = self.uwbt_datarecorder.last_appended_data

        # 最新のデータから処理（最大10個）
        for data in uwbt_data:
            # NLOS判定（1.0がNLOS、0.0がLOS）
            is_nlos = data["nlos"]

            # タイムスタンプベースでタグ位置を取得
            uwb_timestamp = data["sensor_timestamp"]
            tag_loc, tag_quat = self._uwb_get_synchronized_tag_pose(
                data["tag_id"], uwb_timestamp
            )
            if tag_loc is None or tag_quat is None:
                continue

            # 球面座標からローカルデカルト座標への変換
            distance = data["distance"]
            azimuth_rad = np.deg2rad(data["aoa_azimuth"])
            elevation_rad = np.deg2rad(data["aoa_elevation"])

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
            if is_nlos:
                confidence *= 0.8  # NLOSの場合は信頼度を20%低減（以前は50%だった）
            else:
                confidence *= 1.0  # LOSの場合は信頼度を維持

            estimate = TagEstimate(
                tag_id=data["tag_id"],
                position=world_point,
                confidence=confidence,
                distance=distance,
                method="UWBT",
                is_nlos=is_nlos,
            )
            estimates.append(estimate)

        return estimates

    @final
    def __uwb_estimate_from_uwbp(self) -> list[TagEstimate]:
        """UWBPデータから指定タグの位置を推定"""
        estimates = []
        uwbp_data = self.uwbp_datarecorder.last_appended_data

        # 最新のデータから処理（最大10個）
        for data in uwbp_data:
            # タイムスタンプベースでタグ位置を取得
            uwb_timestamp = data["sensor_timestamp"]
            tag_loc, tag_quat = self._uwb_get_synchronized_tag_pose(
                data["tag_id"], uwb_timestamp
            )
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
                tag_id=data["tag_id"],
                position=world_point,
                confidence=confidence,
                distance=distance,
                method="UWBP",
                is_nlos=False,  # UWBPはNLOS情報なし
            )
            estimates.append(estimate)

        return estimates
