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
        scatter = ax.scatter(df.x, df.y, s=3, c=c, label="location (estimated)")
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
            )

        if ground_truth_df is not None:
            ax.scatter(
                ground_truth_df["x"],
                ground_truth_df["y"],
                s=3,
                c="black",
                alpha=0.2,
                label="location (ground truth)",
            )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"UWB Position Estimation - Final Result ({len(df)} points)")
        plt.colorbar(scatter, ax=ax, label="Time step")
        plt.legend()

        if save:
            self.logger.info(f"プロットを {output_file} に保存します")
            plt.savefig(src_dir / output_file)
        if show:
            self.logger.info("プロットを表示します")
            plt.show()
        
        # 各タグの軌跡を個別にプロット
        if hasattr(self, 'get_tag_trajectories'):
            self.plot_tag_trajectories(
                map_file=map_file,
                output_file=output_file,
                map_origin=map_origin,
                map_ppm=map_ppm,
                show=show,
                save=save,
                blue_only=False  # デフォルトは4色表示
            )
    
    def plot_tag_trajectories(
        self,
        map_file: str | Path,
        output_file: str | Path,
        map_origin: tuple[float, float] = (-5.625, -12.75),
        map_ppm: float = 100,
        show: bool = True,
        save: bool = True,
        blue_only: bool = False,
    ) -> None:
        """
        各タグの軌跡を個別のファイルにプロットする（LOS/NLOS情報で色分け）
        
        Args:
            blue_only: Trueの場合、青色（LOS & GPOSに近い）のみを表示
        """
        src_dir = Path().resolve()
        bitmap_array = np.array(Image.open(src_dir / map_file)) / 255.0
        
        height, width = bitmap_array.shape[:2]
        width_m = width / map_ppm
        height_m = height / map_ppm
        
        extent = (
            map_origin[0],
            map_origin[0] + width_m,
            map_origin[1],
            map_origin[1] + height_m,
        )
        
        # LOS情報付きタグ軌跡を優先的に取得
        use_los_info = False
        use_raw_measurements = False
        
        # 生の測定値があるかチェック
        if hasattr(self, 'get_raw_measurements'):
            raw_measurements = self.get_raw_measurements()
            if raw_measurements:
                use_raw_measurements = True
                trajectories_to_plot = raw_measurements
                use_los_info = True  # 生の測定値にはLOS情報が含まれる
        
        if not use_raw_measurements:
            if hasattr(self, 'get_tag_trajectories_with_los'):
                tag_trajectories_with_los = self.get_tag_trajectories_with_los()
                if tag_trajectories_with_los:
                    use_los_info = True
                    trajectories_to_plot = tag_trajectories_with_los
        
        if not use_raw_measurements and not use_los_info:
            # 通常のタグ軌跡を取得（後方互換性）
            tag_trajectories = self.get_tag_trajectories()
            if not tag_trajectories:
                self.logger.info("タグの軌跡データがありません")
                return
            trajectories_to_plot = tag_trajectories
        
        # 各タグごとに個別のプロットを作成
        
        for idx, (tag_id, trajectory) in enumerate(trajectories_to_plot.items()):
            if not trajectory:
                continue
            
            # 新しい図を作成
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.imshow(bitmap_array, extent=extent, alpha=0.5, cmap="gray")
            
            if use_los_info:
                # LOS情報付きの軌跡をDataFrameに変換（GPOSとの距離情報も含む）
                if hasattr(trajectory[0], 'gpos_distance'):
                    tag_df = pd.DataFrame(
                        [(pos.x, pos.y, pos.z, pos.is_los, pos.confidence, pos.method,
                          getattr(pos, 'gpos_distance', 0.0), getattr(pos, 'is_far_from_gpos', False)) 
                         for pos in trajectory],
                        columns=["x", "y", "z", "is_los", "confidence", "method", "gpos_distance", "is_far_from_gpos"]
                    )
                else:
                    tag_df = pd.DataFrame(
                        [(pos.x, pos.y, pos.z, pos.is_los, pos.confidence, pos.method) 
                         for pos in trajectory],
                        columns=["x", "y", "z", "is_los", "confidence", "method"]
                    )
                    tag_df['gpos_distance'] = 0.0
                    tag_df['is_far_from_gpos'] = False
                
                # LOSとNLOSを分離
                los_df = tag_df[tag_df['is_los'] == True]
                nlos_df = tag_df[tag_df['is_los'] == False]
                
                # UWBT と UWBP を分離
                uwbt_df = tag_df[tag_df['method'] == 'UWBT']
                uwbp_df = tag_df[tag_df['method'] == 'UWBP']
                
                # NLOSを含むかどうかをチェック
                has_nlos = len(nlos_df) > 0
                
                # 軌跡の線をプロット（グレー）
                ax.plot(tag_df.x, tag_df.y, '-', 
                       color='gray', alpha=0.3, linewidth=1,
                       label=f"Tag {tag_id} trajectory")
                
                # GPOSとの距離で分離
                close_to_gpos_df = tag_df[tag_df['is_far_from_gpos'] == False]
                far_from_gpos_df = tag_df[tag_df['is_far_from_gpos'] == True]
                
                # デバッグ情報を出力
                print(f"\n=== Tag {tag_id} Debug Info ===")
                print(f"Total points: {len(tag_df)}")
                print(f"Close to GPOS: {len(close_to_gpos_df)}")
                print(f"Far from GPOS: {len(far_from_gpos_df)}")
                if 'gpos_distance' in tag_df.columns:
                    print(f"Distance range: {tag_df['gpos_distance'].min():.2f} - {tag_df['gpos_distance'].max():.2f}m")
                    print(f"Points >3m: {(tag_df['gpos_distance'] >= 3.0).sum()}")
                print("="*30)
                
                if blue_only:
                    # 青色のみ表示（LOS & 3m以内）
                    blue_points = close_to_gpos_df[close_to_gpos_df['is_los'] == True]
                    
                    if not blue_points.empty:
                        ax.scatter(blue_points.x, blue_points.y, s=20, 
                                 color='blue', alpha=0.7, 
                                 edgecolors='darkblue', linewidth=0.5,
                                 label=f"Reliable Points: LOS & <3m from GPOS ({len(blue_points)} points)", zorder=3)
                    
                    # 統計情報を更新（青色のみの場合）
                    filtered_ratio = len(blue_points) / len(tag_df) * 100 if len(tag_df) > 0 else 0
                    
                    # タイトルに情報追加
                    ax.set_title(f"Tag {tag_id} - Reliable Points Only\\n"
                               f"Total: {len(tag_df)} → Filtered: {len(blue_points)} ({filtered_ratio:.1f}%)")
                    
                else:
                    # 4色表示（従来機能）
                    # GPOSに近い点（3m以内）をプロット
                    if not close_to_gpos_df.empty:
                        los_close = close_to_gpos_df[close_to_gpos_df['is_los'] == True]
                        nlos_close = close_to_gpos_df[close_to_gpos_df['is_los'] == False]
                        
                        # LOS & 3m以内 → 青色
                        if not los_close.empty:
                            ax.scatter(los_close.x, los_close.y, s=20, 
                                     color='blue', alpha=0.7, 
                                     edgecolors='darkblue', linewidth=0.5,
                                     label=f"LOS & <3m from GPOS ({len(los_close)} points)", zorder=3)
                        
                        # NLOS & 3m以内 → 黄色
                        if not nlos_close.empty:
                            ax.scatter(nlos_close.x, nlos_close.y, s=25, 
                                     color='yellow', alpha=0.8, 
                                     edgecolors='gold', linewidth=0.5,
                                     marker='s',  # 四角でNLOSを表示
                                     label=f"NLOS & <3m from GPOS ({len(nlos_close)} points)", zorder=3)
                    
                    # GPOSから遠い点（3m以上）をプロット
                    if not far_from_gpos_df.empty:
                        los_far = far_from_gpos_df[far_from_gpos_df['is_los'] == True]
                        nlos_far = far_from_gpos_df[far_from_gpos_df['is_los'] == False]
                        
                        # LOS & 3m以上 → 紫色
                        if not los_far.empty:
                            ax.scatter(los_far.x, los_far.y, s=30, 
                                     color='purple', alpha=0.8, 
                                     edgecolors='darkviolet', linewidth=1.0,
                                     marker='^',  # 三角でGPOSから遠いLOSを表示
                                     label=f"LOS & ≥3m from GPOS ({len(los_far)} points)", zorder=4)
                        
                        # NLOS & 3m以上 → 赤色
                        if not nlos_far.empty:
                            ax.scatter(nlos_far.x, nlos_far.y, s=35, 
                                     color='red', alpha=0.9, 
                                     edgecolors='darkred', linewidth=1.0,
                                     marker='x',  # ×でGPOSから遠いNLOSを強調
                                     label=f"NLOS & ≥3m from GPOS ({len(nlos_far)} points)", zorder=5)
                
                # 統計情報を図に追加
                los_ratio = len(los_df) / len(tag_df) * 100 if len(tag_df) > 0 else 0
                uwbt_count = len(uwbt_df)
                uwbp_count = len(uwbp_df)
                
                info_text = f"Total points: {len(trajectory)}\n"
                info_text += f"LOS: {len(los_df)} ({los_ratio:.1f}%)\n"
                info_text += f"NLOS: {len(nlos_df)} ({100-los_ratio:.1f}%)\n"
                info_text += f"UWBT: {uwbt_count}, UWBP: {uwbp_count}\n"
                info_text += f"Distance traveled: {self._calculate_total_distance(tag_df):.2f} m"
                
            else:
                # 通常の軌跡をDataFrameに変換（後方互換性）
                tag_df = pd.DataFrame(
                    [(pos.x, pos.y, pos.z) for pos in trajectory],
                    columns=["x", "y", "z"]
                )
                
                # 軌跡の線をプロット
                ax.plot(tag_df.x, tag_df.y, '-', 
                       color='blue', alpha=0.7, linewidth=2,
                       label=f"Tag {tag_id} trajectory")
                
                # 各点をプロット
                scatter = ax.scatter(tag_df.x, tag_df.y, 
                                   c=range(len(tag_df)), cmap='viridis',
                                   s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
                
                plt.colorbar(scatter, ax=ax, label="Time step")
                
                # 統計情報を図に追加
                info_text = f"Total points: {len(trajectory)}\n"
                info_text += f"Distance traveled: {self._calculate_total_distance(tag_df):.2f} m"
            
            # 始点を強調（緑の四角）
            ax.scatter(tag_df.x.iloc[0], tag_df.y.iloc[0], 
                     s=200, color='green', marker='s', 
                     edgecolors='black', linewidth=2,
                     label=f"Start", zorder=5)
            
            # 終点を強調（オレンジの三角）
            ax.scatter(tag_df.x.iloc[-1], tag_df.y.iloc[-1], 
                     s=200, color='orange', marker='^', 
                     edgecolors='black', linewidth=2,
                     label=f"End", zorder=5)
            
            # 軸とタイトルの設定
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            
            if use_los_info:
                has_nlos = any(not pos.is_los for pos in trajectory) if hasattr(trajectory[0], 'is_los') else False
                nlos_count = sum(1 for pos in trajectory if not pos.is_los) if hasattr(trajectory[0], 'is_los') else 0
                los_count = len(trajectory) - nlos_count
                if has_nlos:
                    ax.set_title(f"UWB Position Estimation - Tag {tag_id} (LOS: {los_count}, NLOS: {nlos_count})")
                else:
                    ax.set_title(f"UWB Position Estimation - Tag {tag_id} (LOS Only: {los_count} points)")
            else:
                ax.set_title(f"UWB Position Estimation - Tag {tag_id} ({len(trajectory)} points)")
            
            ax.grid(True, alpha=0.3)
            
            # 凡例
            ax.legend(loc='upper right')
            
            # 統計情報をテキストボックスで表示
            ax.text(0.02, 0.98, info_text, 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if save:
                # 出力ファイル名を生成（タグIDを含む）
                output_path = Path(output_file)
                tag_output_file = output_path.parent / f"{output_path.stem}_tag_{tag_id}{output_path.suffix}"
                self.logger.info(f"Tag {tag_id} のプロットを {tag_output_file} に保存します")
                plt.savefig(src_dir / tag_output_file, bbox_inches='tight', dpi=150)
            
            if show:
                plt.show()
            
            plt.close(fig)  # メモリ解放のため図を閉じる
            
            self.logger.info(f"Tag {tag_id}: {len(trajectory)} points plotted")
    
    def _calculate_total_distance(self, df: pd.DataFrame) -> float:
        """軌跡の総移動距離を計算"""
        if len(df) < 2:
            return 0.0
        
        distances = np.sqrt(
            np.diff(df.x)**2 + 
            np.diff(df.y)**2 + 
            np.diff(df.z)**2
        )
        return np.sum(distances)
