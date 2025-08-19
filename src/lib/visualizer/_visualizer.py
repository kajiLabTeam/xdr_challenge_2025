import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from src.lib.groundtruth._groundtruth import GroundTruth
from src.lib.recorder._recorder import DataRecorderProtocol


class Visualizer(DataRecorderProtocol):
    """
    データの可視化を行うクラス
    """

    def plot_map(
        self,
        map_file: str | Path,
        output_file: str | Path = "output_map.png",
        map_origin: tuple[float, float] = (-5.625, -12.75),
        map_ppm: float = 100,
        show: bool = True,
        save: bool = True,
        maxwait: float | None = None,
        gpos: bool = False,
        ground_truth_df: pd.DataFrame | None = None,
    ) -> None:
        """
        推定結果をプロットする

        Args:
            map_file (str | Path): マップ画像ファイルのパス
            output_file (str | Path | None): 出力ファイルのパス。
            map_origin (tuple[float, float]): マップの原点座標 (x, y)
            map_ppm (float): ピクセルあたりのメートル数
            show (bool): プロットを表示するかどうか
            save (bool): プロットをファイルに保存するかどうか
        """
        self.logger.info("推定結果をマップにプロットします")

        src_dir = Path().resolve()
        bitmap_array = np.array(Image.open(src_dir / map_file)) / 255.0
        df = pd.DataFrame(self.positions, columns=["x", "y", "z"])

        height, width = bitmap_array.shape[:2]
        width_m = width / map_ppm
        height_m = height / map_ppm

        extent = (
            map_origin[0],
            map_origin[0] + width_m,
            map_origin[1],
            map_origin[1] + height_m,
        )

        # メインプロット（最終推定結果）
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.imshow(bitmap_array, extent=extent, alpha=0.5, cmap="gray")
        c = df.index * maxwait if maxwait else df.index
        scatter = ax.scatter(
            df.x, df.y, s=3, c=c, label="location (estimated)", zorder=100
        )
        if gpos:
            gpos_data = self.gpos_datarecorder.data
            gpos_df = pd.DataFrame(gpos_data)
            ax.scatter(
                gpos_df["location_x"],
                gpos_df["location_y"],
                s=3,
                c="red",
                alpha=0.2,
                label="location (GPOS)",
                zorder=0,
            )

        if ground_truth_df is not None:
            ax.scatter(
                ground_truth_df["x"],
                ground_truth_df["y"],
                s=3,
                c="black",
                alpha=0.2,
                label="location (ground truth)",
                zorder=50,
            )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.colorbar(scatter, ax=ax, label="timestamp (s)")
        plt.legend()

        if save:
            self.logger.info(f"プロットを {output_file} に保存します")
            plt.savefig(src_dir / output_file)
        if show:
            self.logger.info("プロットを表示します")
            plt.show()
