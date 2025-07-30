import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from src.lib.recorder._recorder import DataRecorderProtocol


class Visualizer(DataRecorderProtocol):
    """
    データの可視化を行うクラス
    """

    def plot_map(
        self,
        map_file: str | Path,
        output_file: str | Path | None = None,
        map_origin: tuple[float, float] = (-5.625, -12.75),
        map_ppm: float = 100,
        show: bool = True,
        save: bool = True,
    ) -> None:
        """
        推定結果をプロットする

        Args:
            filename: 保存するファイル名。Noneの場合は表示する
        """
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

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.imshow(bitmap_array, extent=extent, alpha=0.5, cmap="gray")
        scatter = ax.scatter(df.x, df.y, s=3, label="location (ground truth)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        plt.colorbar(scatter, ax=ax, label="timestamp (s)")
        plt.legend()

        if output_file:
            plt.savefig(src_dir / output_file)

        plt.show()
