from logging import Logger
from pathlib import Path
from typing import final, Optional
from PIL import Image
from collections import deque
import numpy as np
import pandas as pd
from src.lib.params._params import Params
from src.lib.visualizer._visualizer import VisualizerProtocol
from src.type import Pixel, Position, TimedPosition, TimedPose


class MapMatching(VisualizerProtocol):

    MAP_RESOLUTION: float = 0.01
    MAP_ORIGIN_X_IN_PIXELS: int = 565
    MAP_ORIGIN_Y_IN_PIXELS: int = 1480
    WALKABLE_THRESHOLD: int = 20

    EXHIBITS = [  # 展示物の位置と向きの定義 でもorientationが機能してないかも（サンプルデータに向き情報がないため）
        {"id": "exhibit_A", "pos": Position(19.5, -10.0, 0), "orientation": np.pi},
        {"id": "exhibit_B", "pos": Position(14.5, -10.0, 0), "orientation": np.pi},
        {"id": "exhibit_C", "pos": Position(11.5, -6.7, 0), "orientation": -np.pi / 2},
    ]
    """
    展示物のリスト。各展示物はID、実世界座標(pos)、および向き(orientation)を持つ。
    """

    def __init__(self, trial_id: str, logger: Logger) -> None:
        """
        MapMatchingクラスのインスタンスを初期化する。
        Args:
            trial_id (str): 試行ID。
            logger (Logger): ロギング用のロガーインスタンス。
        """
        self.map_file = Path() / "map" / "miraikan_5_custom.png"
        self.logger = logger

        # 軌跡全体のオフセットを管理する変数
        self.trajectory_offset = (0.0, 0.0)
        self.last_known_angle: Optional[float] = (
            None  # 最後に知られている向き(ラジアン)
        )

        try:
            self.map_image = Image.open(self.map_file).convert("L")
        except FileNotFoundError:
            self.logger.error(f"マップファイルが見つかりません: {self.map_file}")

    def map_matching(self, poses: list[TimedPose]) -> list[TimedPose]:
        """
        マップマッチングを実行する

        Args:
            poses (list[TimedPose]): 入力の位置リスト。
        Returns:
            list[TimedPose]: マップマッチング後の位置リスト。
        """
        pose_df = pd.DataFrame(
            poses,
            columns=["x", "y", "z", "yaw", "timestamp"],
        )
        # timestampの差を出す
        pose_df["dt"] = pose_df["timestamp"].diff().fillna(0)
        # Params.stop_walk_sec() 秒以上の差がある場合、停止とみなす
        pose_df["is_stopped"] = pose_df["dt"] > Params.stop_walk_sec()

        for i in range(len(pose_df)):
            current_series = pose_df.iloc[i]

            if current_series["is_stopped"]:
                # 展示物
                pass

        return []  # TODO

    def _find_nearest_exhibit(self, position: Position) -> Optional[dict]:
        """
        指定された位置から最も近い展示物を見つける。
        Args:
            position (Position): 現在の位置。
        Returns:
            Optional[dict]: 最も近い展示物の情報。展示物リストが空の場合はNone。
        """
        if not self.EXHIBITS:
            return None
        ##--- 展示物の位置がPositionオブジェクトであることを確認し、距離を計算 ---##
        distances = [
            np.linalg.norm([position.x - ex["pos"].x, position.y - ex["pos"].y])
            for ex in self.EXHIBITS
            if hasattr(ex["pos"], "x") and hasattr(ex["pos"], "y")
        ]
        return self.EXHIBITS[np.argmin(distances)]

    def _is_path_clear(self, p1: Pixel, p2: Pixel) -> bool:
        """
        2つのピクセル座標間に障害物（通行不可能なピクセル）がないかどうかを判定する。
        ブレゼンハムのアルゴリズムを使用。
        Args:
            p1 (Pixel): 始点のピクセル座標。
            p2 (Pixel): 終点のピクセル座標。
        Returns:
            bool: 経路が通行可能であればTrue、そうでなければFalse。
        """
        dx = abs(p2.x - p1.x)
        dy = -abs(p2.y - p1.y)
        sx = 1 if p1.x < p2.x else -1
        sy = 1 if p1.y < p2.y else -1
        err = dx + dy
        temp_px, temp_py = p1.x, p1.y
        while True:
            if not self._is_walkable(Pixel(temp_px, temp_py)):
                return False
            if temp_px == p2.x and temp_py == p2.y:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                temp_px += sx
            if e2 <= dx:
                err += dx
                temp_py += sy
        return True

    def _find_collision_point(self, p1: Pixel, p2: Pixel) -> Pixel:
        """
        2つのピクセル座標を結ぶ直線上で、最初に見つかる通行不可能なピクセル（衝突点）を返す。
        Args:
            p1 (Pixel): 始点のピクセル座標。
            p2 (Pixel): 終点のピクセル座標。
        Returns:
            Pixel: 最初に検出された通行不可能なピクセルの座標。衝突がなければp2を返す。
        """
        dx = abs(p2.x - p1.x)
        dy = -abs(p2.y - p1.y)
        sx = 1 if p1.x < p2.x else -1
        sy = 1 if p1.y < p2.y else -1
        err = dx + dy
        temp_px, temp_py = p1.x, p1.y
        while True:
            if not self._is_walkable(Pixel(temp_px, temp_py)):
                return Pixel(temp_px, temp_py)
            if temp_px == p2.x and temp_py == p2.y:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                temp_px += sx
            if e2 <= dx:
                err += dx
                temp_py += sy
        return p2

    def _get_corridor_direction(
        self, pixel: Pixel, search_dist: int = 5
    ) -> Optional[str]:
        """
        指定されたピクセル周辺の通行可能な方向を調べ、
        縦方向(N-S)または横方向(E-W)のどちらかが明確に多い場合にその方向を返す。
        どちらの方向も明確でない場合はNoneを返す。
        Args:
            pixel (Pixel): 調査対象のピクセル
            search_dist (int): 調査する距離(ピクセル単位)
        Returns:
            Optional[str]: 'NS', 'EW', または None
        """
        counts = {
            "N": 0,
            "S": 0,
            "E": 0,
            "W": 0,
        }  # 各方向の通行可能なピクセル数をカウント
        for i in range(1, search_dist + 1):  # 上方向
            if self._is_walkable(Pixel(pixel.x, pixel.y - i)):
                counts["N"] += 1
            else:
                break
        for i in range(1, search_dist + 1):  # 下方向
            if self._is_walkable(Pixel(pixel.x, pixel.y + i)):
                counts["S"] += 1
            else:
                break
        for i in range(1, search_dist + 1):  # 右方向
            if self._is_walkable(Pixel(pixel.x + i, pixel.y)):
                counts["E"] += 1
            else:
                break
        for i in range(1, search_dist + 1):  # 左方向
            if self._is_walkable(Pixel(pixel.x - i, pixel.y)):
                counts["W"] += 1
            else:
                break
        vertical_walkable = counts["N"] + counts["S"]  # 縦方向の通行可能なピクセル数
        horizontal_walkable = counts["E"] + counts["W"]  # 横方向の通行可能なピクセル数
        if vertical_walkable > horizontal_walkable * 1.5:  # 明確に縦方向が多い場合
            return "NS"
        if horizontal_walkable > vertical_walkable * 1.5:  # 明確に横方向が多い場合
            return "EW"
        return None

    def _find_nearest_passable_point(self, start_pixel: Pixel) -> Pixel:
        """
        指定されたピクセルから最も近い通行可能なピクセルを探索する
        Args:
            start_pixel (Pixel): 探索開始ピクセル
        Returns:
            Pixel: 最も近い通行可能なピクセル
        """
        queue = deque([start_pixel])  # BFS用のキュー
        visited = {start_pixel}  # 訪問済みピクセルの集合
        while queue:
            current_pixel = queue.popleft()  # キューからピクセルを取り出す
            if self._is_walkable(current_pixel):  # 通行可能ならそのピクセルを返す
                return current_pixel
            for dx in [-1, 0, 1]:  # 周囲8方向を探索
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    next_pixel = Pixel(current_pixel.x + dx, current_pixel.y + dy)  #
                    if next_pixel not in visited:
                        visited.add(next_pixel)
                        queue.append(next_pixel)
        return start_pixel

    @final
    def _world_to_pixel(self, position: Position | TimedPosition) -> Pixel:
        """
        実世界座標(メートル)をピクセル座標に変換する
        Args:
            position (Position): 実世界座標
        Returns:
            Pixel: ピクセル座標
        """

        pixel_x = int(self.MAP_ORIGIN_X_IN_PIXELS + position.x / self.MAP_RESOLUTION)
        pixel_y = int(self.MAP_ORIGIN_Y_IN_PIXELS - position.y / self.MAP_RESOLUTION)

        return Pixel(pixel_x, pixel_y)

    @final
    def _pixel_to_world(self, pixel: Pixel) -> Position:
        """
        ピクセル座標を実世界座標(メートル)に変換する
        Args:
            pixel (Pixel): ピクセル座標
        Returns:
            Position: 実世界座標(メートル)
        """
        x = (pixel.x - self.MAP_ORIGIN_X_IN_PIXELS) * self.MAP_RESOLUTION
        y = (self.MAP_ORIGIN_Y_IN_PIXELS - pixel.y) * self.MAP_RESOLUTION

        return Position(x, y, 0.0)

    @final
    def _is_walkable(self, pixel: Pixel) -> bool:
        """
        指定されたピクセルが通行可能かどうかを返す
        Args:
            pixel (Pixel): ピクセル座標
        Returns:
            bool: 通行可能か
        """
        width, height = self.map_image.size
        if not (0 <= pixel.x < width and 0 <= pixel.y < height):
            return False

        gray = self.map_image.getpixel((pixel.x, pixel.y))

        if not (isinstance(gray, float) or isinstance(gray, int)):
            raise TypeError("Gray value is not a float or int")

        return gray > self.WALKABLE_THRESHOLD
