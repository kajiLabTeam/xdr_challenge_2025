from logging import Logger
from pathlib import Path
from typing import final
from PIL import Image

from src.lib.visualizer._visualizer import VisualizerProtocol
from src.type import Pixel, Position, PositionWithTimestamp


class MapMatching(VisualizerProtocol):

    MAP_RESOLUTION: float = 0.01
    MAP_ORIGIN_X_IN_PIXELS: int = 565
    MAP_ORIGIN_Y_IN_PIXELS: int = 1480
    WALKABLE_THRESHOLD: int = 200

    def __init__(self, trial_id: str, logger: Logger) -> None:
        self.map_file = Path() / "map" / "miraikan_5_custom.png"
        self.logger = logger

        try:
            self.map_image = Image.open(self.map_file).convert("L")
        except FileNotFoundError:
            self.logger.error(f"マップファイルが見つかりません: {self.map_file}")

    def map_matching(self, positions: list[PositionWithTimestamp]) -> list[Position]:
        """
        マップマッチング
        Args:
            positions (list[PositionWithTimestamp]): 実世界座標のリスト
        Returns:
            list[Position]: マップ上のピクセル座標のリスト
        """
        pixels = list(map(self._world_to_pixel, positions))
        # 未確定の軌跡
        unconfirmed_trajectory: list[Pixel] = []
        # 確定した軌跡
        confirmed_trajectory: list[Pixel] = []

        for i, pixel in enumerate(pixels):
            unconfirmed_trajectory.append(pixel)

            # if self._is_reference_position(i, pixels):
            #    confirmed_trajectory.extend(unconfirmed_trajectory)
            #    unconfirmed_trajectory.clear()

        return list(map(self._pixel_to_world, confirmed_trajectory))

    def _is_reference_position(self, target_index: int, pixels: list[Pixel]) -> bool:
        """
        基準位置かどうかを判定する
        Args:
            target_index (int): 判定対象のインデックス
            pixels (list[Pixel]): ピクセル座標のリスト
        Returns:
            bool: 基準位置かどうか
        """
        # 基準位置の判定ロジックを実装
        raise NotImplementedError(
            "基準位置の判定ロジックが未実装です。_is_reference_position"
        )

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
