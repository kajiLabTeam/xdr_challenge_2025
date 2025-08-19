#!/usr/bin/env python
"""
UWBTのみのデータで各タグの軌跡をプロット
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.lib import Localizer
import logging

def plot_uwbt_trajectories():
    """UWBTのみの軌跡をプロット"""
    
    # ロガー設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Localizerを初期化して実行
    logger.info("デモモードでLocalizerを実行中...")
    localizer = Localizer("demo", logger)
    
    # 簡易的にデータを読み込む（デモモード）
    from src.lib.requester import ImmediateRequester
    requester = ImmediateRequester("", "demo", logger)
    
    # 初期状態を設定
    initial_state = requester.send_state_req()
    if initial_state:
        localizer.set_init_pos(initial_state.pos)
    
    # 全データを処理
    while True:
        recv_data = requester.send_nextdata_req(position=localizer[-1])
        
        if recv_data is None:
            break
            
        from src.type import SensorData
        if isinstance(recv_data, SensorData):
            localizer.clear_last_appended_data()
            localizer.set_sensor_data(recv_data)
            localizer.estimate()
        else:
            break
    
    # UWBTのみのデータを取得
    uwbt_only_measurements = localizer.get_uwbt_only_measurements()
    
    if not uwbt_only_measurements:
        logger.error("UWBTデータがありません")
        return
    
    # マップの読み込み
    map_file = "map/miraikan_5.bmp"
    map_ppm = 30.0
    map_origin = (0.0, 0.0)
    
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
    
    # 出力ディレクトリの作成
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 各タグごとにプロット
    for tag_id, measurements in uwbt_only_measurements.items():
        if not measurements:
            continue
        
        # DataFrameに変換
        df = pd.DataFrame(
            [(pos.x, pos.y, pos.z, pos.is_los, pos.confidence) 
             for pos in measurements],
            columns=["x", "y", "z", "is_los", "confidence"]
        )
        
        # LOSとNLOSを分離
        los_df = df[df['is_los'] == True]
        nlos_df = df[df['is_los'] == False]
        
        # 統計情報
        total_points = len(df)
        los_count = len(los_df)
        nlos_count = len(nlos_df)
        nlos_ratio = (nlos_count / total_points * 100) if total_points > 0 else 0
        
        # プロット作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # 左側: 軌跡プロット
        ax1.imshow(bitmap_array, extent=extent, alpha=0.5, cmap="gray")
        
        # 軌跡の線をプロット（時系列で色を変える）
        scatter = ax1.scatter(df.x, df.y, 
                            c=range(len(df)), cmap='viridis',
                            s=5, alpha=0.6)
        
        # LOSとNLOSを重ねてプロット
        if not los_df.empty:
            ax1.scatter(los_df.x, los_df.y, s=20, 
                       color='blue', alpha=0.6, 
                       edgecolors='darkblue', linewidth=0.5,
                       label=f"LOS ({los_count} points)", zorder=3)
        
        if not nlos_df.empty:
            ax1.scatter(nlos_df.x, nlos_df.y, s=30, 
                       color='red', alpha=0.8, 
                       edgecolors='darkred', linewidth=0.5,
                       marker='x',
                       label=f"NLOS ({nlos_count} points)", zorder=4)
        
        # 開始点と終了点をマーク
        ax1.scatter(df.x.iloc[0], df.y.iloc[0], s=200, 
                   color='green', marker='o', edgecolors='black', 
                   linewidth=2, label='Start', zorder=5)
        ax1.scatter(df.x.iloc[-1], df.y.iloc[-1], s=200, 
                   color='red', marker='s', edgecolors='black', 
                   linewidth=2, label='End', zorder=5)
        
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title(f"Tag {tag_id} - UWBT Only Trajectory")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # カラーバー追加
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Time Index')
        
        # 右側: 時系列プロット
        time_index = range(len(df))
        
        # 距離の時系列
        ax2.scatter(time_index, df.index.map(lambda i: np.sqrt((df.x.iloc[i] - df.x.iloc[0])**2 + 
                                                               (df.y.iloc[i] - df.y.iloc[0])**2)),
                   c=df['is_los'].map({True: 'blue', False: 'red'}),
                   s=10, alpha=0.6)
        
        ax2.set_xlabel("Time Index")
        ax2.set_ylabel("Distance from Start (m)")
        ax2.set_title(f"Distance from Start Over Time")
        ax2.grid(True, alpha=0.3)
        
        # NLOS期間を背景色で表示
        for i in range(len(df) - 1):
            if not df.iloc[i]['is_los']:
                ax2.axvspan(i, i+1, alpha=0.2, color='red')
        
        # 統計情報を追加
        stats_text = f"Total UWBT points: {total_points}\n"
        stats_text += f"LOS: {los_count} ({100-nlos_ratio:.1f}%)\n"
        stats_text += f"NLOS: {nlos_count} ({nlos_ratio:.1f}%)\n"
        stats_text += f"Mean confidence: {df['confidence'].mean():.3f}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.suptitle(f"Tag {tag_id} - UWBT Only Analysis", fontsize=14)
        plt.tight_layout()
        
        # 保存
        output_file = output_dir / f"uwbt_only_tag_{tag_id}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Tag {tag_id} のUWBTのみの軌跡を {output_file} に保存しました")
        logger.info(f"  - Total: {total_points}, LOS: {los_count}, NLOS: {nlos_count} ({nlos_ratio:.1f}%)")
        
        # CSVファイルも保存
        csv_file = output_dir / f"uwbt_only_tag_{tag_id}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"  - CSVファイル: {csv_file}")
        
        plt.show()
        plt.close()


if __name__ == "__main__":
    plot_uwbt_trajectories()