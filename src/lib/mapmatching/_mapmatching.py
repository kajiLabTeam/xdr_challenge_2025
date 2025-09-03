from logging import Logger
from pathlib import Path
from typing import final, Optional
from PIL import Image
from collections import deque
import numpy as np

from src.lib.visualizer._visualizer import VisualizerProtocol
from src.type import Pixel, Position, PositionWithTimestamp


class MapMatching(VisualizerProtocol):

    MAP_RESOLUTION: float = 0.01
    MAP_ORIGIN_X_IN_PIXELS: int = 565
    MAP_ORIGIN_Y_IN_PIXELS: int = 1480
    WALKABLE_THRESHOLD: int = 200

    EXHIBITS = [# 展示物の位置と向きの定義 でもorientationが機能してないかも（サンプルデータに向き情報がないため）
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
        self.last_known_angle: Optional[float] = None # 最後に知られている向き(ラジアン)

        try:
            self.map_image = Image.open(self.map_file).convert("L")
        except FileNotFoundError:
            self.logger.error(f"マップファイルが見つかりません: {self.map_file}")
    
    def _find_nearest_exhibit(self, position: Position) -> Optional[dict]:
        """
        指定された位置から最も近い展示物を見つける。
        Args:
            position (Position): 現在の位置。
        Returns:
            Optional[dict]: 最も近い展示物の情報。展示物リストが空の場合はNone。
        """
        if not self.EXHIBITS: return None
        ##--- 展示物の位置がPositionオブジェクトであることを確認し、距離を計算 ---##
        distances = [np.linalg.norm([position.x - ex["pos"].x, position.y - ex["pos"].y]) for ex in self.EXHIBITS if hasattr(ex["pos"], "x") and hasattr(ex["pos"], "y")]
        return self.EXHIBITS[np.argmin(distances)]
    
    def map_matching(self, positions: list[PositionWithTimestamp]) -> list[Position]:
        """
        タイムスタンプ付きの位置情報のリストを受け取り、マップマッチングを適用して補正された位置情報のリストを返す。
        停止検出による展示物へのスナップ、壁衝突時の経路補正などを行う。
        Args:
            positions (list[PositionWithTimestamp]): PDRによって推定されたタイムスタンプ付きの位置のリスト。
        Returns:
            list[Position]: マップマッチングによって補正された位置のリスト。
        """

        if not positions:
            return []
            
        corrected_positions = [Position(positions[0].x, positions[0].y, positions[0].z)]

        for i in range(1, len(positions)):
            # --- PDRの予測位置に、これまでの累積オフセットを適用 ---
            # PDRの生の位置
            current_pdr_pos_raw = Position(positions[i].x, positions[i].y, positions[i].z)
            # オフセットを適用した現在のPDR位置
            current_pdr_pos = Position(
                current_pdr_pos_raw.x + self.trajectory_offset[0],
                current_pdr_pos_raw.y + self.trajectory_offset[1],
                current_pdr_pos_raw.z
            )
            
            time_diff = positions[i].timestamp - positions[i-1].timestamp# タイムスタンプの差
            
            # --- 6秒以上の停止を検知した場合の処理 ---
            if time_diff >= 6.0:
                print(f"タイムスタンプの差が6秒以上あります: index={i}, diff={time_diff:.2f}。展示物での停止と判断します。")
                last_pos_before_stop = corrected_positions[-1]
                
                nearest_exhibit = self._find_nearest_exhibit(last_pos_before_stop)
                
                if nearest_exhibit:
                    corrected_pos = nearest_exhibit["pos"]
                    
                    # ★★★ オフセットを計算し、累積更新 ★★★
                    offset_x = corrected_pos.x - current_pdr_pos.x
                    offset_y = corrected_pos.y - current_pdr_pos.y
                    self.trajectory_offset = (self.trajectory_offset[0] + offset_x, self.trajectory_offset[1] + offset_y)
                    # ★★★★★★★★★★★★★★★★★★★★★
                    
                    # 補正後の位置は、現在のPDR位置にオフセットを加算したもの、つまり展示物の位置そのもの
                    final_corrected_pos = Position(
                        current_pdr_pos.x + offset_x,
                        current_pdr_pos.y + offset_y,
                        current_pdr_pos.z
                    )

                    self.last_known_angle = nearest_exhibit["orientation"]# 最後に知られている向きを更新
                    corrected_positions.append(final_corrected_pos)# 補正後の位置を追加
                    
                    print(f"  -> {nearest_exhibit['id']} の位置にスナップ。軌跡全体のオフセットを更新: (dx={self.trajectory_offset[0]:.2f}, dy={self.trajectory_offset[1]:.2f})")
                    continue
            
            # --- 通常のマップマッチング処理 (オフセット適用済みの'current_pdr_pos'を使用) ---
            last_corrected_pos = corrected_positions[-1]# 最後に補正された位置
            current_pdr_px = self._world_to_pixel(current_pdr_pos)
            last_corrected_px = self._world_to_pixel(last_corrected_pos)

            if self._is_path_clear(last_corrected_px, current_pdr_px):
                corrected_positions.append(current_pdr_pos)
            else: # 壁に衝突している場合の処理
                collision_px = self._find_collision_point(last_corrected_px, current_pdr_px)# 衝突点を探索
                snapped_px = self._find_nearest_passable_point(collision_px)# 衝突点から最も近い通行可能なピクセルを探索
                print(f"壁に衝突しました。 衝突点: ({collision_px.x}, {collision_px.y})")
                print(f"衝突点から最も近い通行可能なピクセル: ({snapped_px.x}, {snapped_px.y})")
                
                corridor_direction = self._get_corridor_direction(snapped_px)# 衝突点周辺の通行可能な方向を調査
                movement_vector = (current_pdr_pos.x - last_corrected_pos.x, current_pdr_pos.y - last_corrected_pos.y)# 移動ベクトル
                projected_vector = (0.0, 0.0)# 衝突点からの投影ベクトル
                if corridor_direction == 'NS':
                    projected_vector = (0, movement_vector[1])# 縦方向に投影
                elif corridor_direction == 'EW':
                    projected_vector = (movement_vector[0], 0)# 横方向に投影
                
                new_pos = Position(last_corrected_pos.x + projected_vector[0], last_corrected_pos.y + projected_vector[1], current_pdr_pos.z)# 投影ベクトルを適用した新しい位置
                new_pos_px = self._world_to_pixel(new_pos)

                if projected_vector != (0.0, 0.0) and self._is_walkable(new_pos_px) and self._is_path_clear(last_corrected_px, new_pos_px):#
                    corrected_positions.append(new_pos)
                else:# 投影ベクトルがゼロベクトル、または投影先が通行不可能、または経路が通行不可能な場合
                    if self._is_path_clear(last_corrected_px, snapped_px):# スナップ先までの経路が通行可能ならスナップ先に移動
                        snapped_world_pos = self._pixel_to_world(snapped_px)
                        corrected_positions.append(Position(snapped_world_pos.x, snapped_world_pos.y, current_pdr_pos.z))
                    else:# それ以外は最後に補正された位置を維持
                        corrected_positions.append(last_corrected_pos)
                    
        return corrected_positions

    
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

    def _get_corridor_direction(self, pixel: Pixel, search_dist: int = 5) -> Optional[str]:
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
        counts = {'N': 0, 'S': 0, 'E': 0, 'W': 0}# 各方向の通行可能なピクセル数をカウント
        for i in range(1, search_dist + 1):# 上方向
            if self._is_walkable(Pixel(pixel.x, pixel.y - i)): counts['N'] += 1
            else: break
        for i in range(1, search_dist + 1):# 下方向
            if self._is_walkable(Pixel(pixel.x, pixel.y + i)): counts['S'] += 1
            else: break
        for i in range(1, search_dist + 1):# 右方向
            if self._is_walkable(Pixel(pixel.x + i, pixel.y)): counts['E'] += 1
            else: break
        for i in range(1, search_dist + 1):# 左方向
            if self._is_walkable(Pixel(pixel.x - i, pixel.y)): counts['W'] += 1
            else: break
        vertical_walkable = counts['N'] + counts['S']# 縦方向の通行可能なピクセル数
        horizontal_walkable = counts['E'] + counts['W']# 横方向の通行可能なピクセル数
        if vertical_walkable > horizontal_walkable * 1.5:# 明確に縦方向が多い場合
            return 'NS'
        if horizontal_walkable > vertical_walkable * 1.5:# 明確に横方向が多い場合
            return 'EW'
        return None

    def _find_nearest_passable_point(self, start_pixel: Pixel) -> Pixel:
        """
        指定されたピクセルから最も近い通行可能なピクセルを探索する
        Args:
            start_pixel (Pixel): 探索開始ピクセル
        Returns:
            Pixel: 最も近い通行可能なピクセル
        """
        queue = deque([start_pixel])# BFS用のキュー
        visited = {start_pixel}# 訪問済みピクセルの集合
        while queue:
            current_pixel = queue.popleft()# キューからピクセルを取り出す
            if self._is_walkable(current_pixel):# 通行可能ならそのピクセルを返す
                return current_pixel
            for dx in [-1, 0, 1]:# 周囲8方向を探索
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    next_pixel = Pixel(current_pixel.x + dx, current_pixel.y + dy)#
                    if next_pixel not in visited:
                        visited.add(next_pixel)
                        queue.append(next_pixel)
        return start_pixel

    @final
    def _world_to_pixel(self, position: Position | PositionWithTimestamp) -> Pixel:
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
