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
        df = pd.DataFrame(self.poses, columns=["x", "y", "z", "yaw", "timestamp"])

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
        scatter = ax.scatter(df.x, df.y, s=3, c=c, label="estimated", zorder=100)
        if gpos:
            gpos_data = self.gpos_datarecorder.data
            gpos_df = pd.DataFrame(gpos_data)
            ax.scatter(
                gpos_df["location_x"],
                gpos_df["location_y"],
                s=1,
                c="red",
                alpha=0.2,
                label="GPOS",
                zorder=0,
            )

        if ground_truth_df is not None:
            ax.scatter(
                ground_truth_df["x"],
                ground_truth_df["y"],
                s=3,
                c="black",
                alpha=0.2,
                label="ground truth",
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

    def plot_yaw_comparison(
        self,
        output_file: str | Path = "yaw_comparison.png",
        ground_truth_df: pd.DataFrame | None = None,
        show: bool = True,
        save: bool = True,
    ) -> None:
        """
        推定されたyaw角とground truthのyaw角を比較するグラフをプロット

        Args:
            output_file (str | Path): 出力ファイルのパス
            ground_truth_df (pd.DataFrame | None): ground truthデータフレーム
            show (bool): プロットを表示するかどうか
            save (bool): プロットをファイルに保存するかどうか
        """
        self.logger.info("yaw角の比較グラフを作成します")
        
        # 推定結果のデータフレーム作成
        estimated_df = pd.DataFrame(self.poses, columns=["x", "y", "z", "yaw", "timestamp"])
        
        # Ground truthデータからyaw角を計算
        if ground_truth_df is not None and "qw" in ground_truth_df.columns:
            # クォータニオンからyaw角を計算
            from scipy.spatial.transform import Rotation as R
            
            yaw_list = []
            for idx, row in ground_truth_df.iterrows():
                if all(col in row for col in ["qw", "qx", "qy", "qz"]):
                    quat = [row["qx"], row["qy"], row["qz"], row["qw"]]
                    rotation = R.from_quat(quat)
                    euler = rotation.as_euler('xyz', degrees=True)
                    yaw_list.append(euler[2])  # z軸周りの回転がyaw
                else:
                    yaw_list.append(0.0)
            ground_truth_df["yaw"] = yaw_list
        
        # グラフの作成
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 上段: yaw角の時系列比較
        ax1 = axes[0]
        ax1.plot(estimated_df["timestamp"], np.degrees(estimated_df["yaw"]), 
                label="Estimated (UWB)", color="blue", linewidth=2, alpha=0.8)
        
        if ground_truth_df is not None and "yaw" in ground_truth_df.columns:
            # Ground truthのタイムスタンプと推定値のタイムスタンプを合わせる
            if "timestamp" in ground_truth_df.columns:
                ax1.plot(ground_truth_df["timestamp"], ground_truth_df["yaw"], 
                        label="Ground Truth", color="red", linewidth=2, alpha=0.8)
            else:
                # タイムスタンプがない場合はインデックスで描画
                ax1.plot(ground_truth_df["yaw"], 
                        label="Ground Truth", color="red", linewidth=2, alpha=0.8)
        
        ax1.set_xlabel("Timestamp (s)")
        ax1.set_ylabel("Yaw Angle (degrees)")
        ax1.set_title("Yaw Angle Comparison: UWB Estimation vs Ground Truth")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        
        # 下段: yaw角の誤差
        ax2 = axes[1]
        if ground_truth_df is not None and "yaw" in ground_truth_df.columns:
            # 最も近いタイムスタンプでマッチング
            errors = []
            timestamps = []
            
            for idx, row in estimated_df.iterrows():
                est_time = row["timestamp"]
                est_yaw = np.degrees(row["yaw"])
                
                # Ground truthで最も近いタイムスタンプを見つける
                if "timestamp" in ground_truth_df.columns:
                    time_diffs = np.abs(ground_truth_df["timestamp"] - est_time)
                    nearest_idx = time_diffs.idxmin()
                    gt_yaw = ground_truth_df.loc[nearest_idx, "yaw"]
                else:
                    # タイムスタンプがない場合はインデックスでマッチング
                    if idx < len(ground_truth_df):
                        gt_yaw = ground_truth_df.iloc[idx]["yaw"]
                    else:
                        continue
                
                # 角度差を-180〜180の範囲に正規化
                error = est_yaw - gt_yaw
                error = np.degrees(np.arctan2(np.sin(np.radians(error)), 
                                              np.cos(np.radians(error))))
                errors.append(error)
                timestamps.append(est_time)
            
            ax2.plot(timestamps, errors, color="green", linewidth=1.5, alpha=0.8)
            ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            
            # 統計情報を追加
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            rmse = np.sqrt(np.mean(np.array(errors) ** 2))
            
            stats_text = f"Mean Error: {mean_error:.2f}°\nStd Dev: {std_error:.2f}°\nRMSE: {rmse:.2f}°"
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        else:
            ax2.text(0.5, 0.5, "Ground Truth data not available", 
                    transform=ax2.transAxes, ha="center", va="center")
        
        ax2.set_xlabel("Timestamp (s)")
        ax2.set_ylabel("Yaw Error (degrees)")
        ax2.set_title("Yaw Angle Error (Estimated - Ground Truth)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存と表示
        src_dir = Path().resolve()
        if save:
            self.logger.info(f"yaw角比較グラフを {output_file} に保存します")
            plt.savefig(src_dir / output_file, dpi=150, bbox_inches="tight")
        if show:
            self.logger.info("yaw角比較グラフを表示します")
            plt.show()
