#!/usr/bin/env python3
"""
UWB推定軌跡プロットスクリプト

このスクリプトは、UWBローカライザーによる推定軌跡をプロットします。
- 各タグの推定軌跡
- 各タグの推定軌跡のLOS/NLOS状態
- 各タグの推定軌跡の進行方向
- Ground truth

使用方法:
    python plot_UWB.py [--output-dir OUTPUT_DIR] [--map-file MAP_FILE]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image
import os
import glob
from typing import Dict, Tuple, Optional
import argparse


def load_map_image(path: str) -> np.ndarray:
    """マップ画像を読み込む"""
    return np.array(Image.open(path)) / 255.0


def plot_map(ax: Axes, bitmap: np.ndarray, map_origin: Tuple[float, float], map_ppm: float) -> None:
    """地図を描画"""
    height, width = bitmap.shape[:2]
    extent: tuple[float, float, float, float] = (
        float(map_origin[0]), 
        float(map_origin[0] + width / map_ppm),
        float(map_origin[1]), 
        float(map_origin[1] + height / map_ppm)
    )
    ax.imshow(bitmap, extent=extent, alpha=0.5, cmap='gray')


def calculate_direction(df: pd.DataFrame) -> pd.DataFrame:
    """進行方向を計算"""
    df = df.copy()
    
    # 位置の差分から進行方向を計算
    dx = df['x'].diff()
    dy = df['y'].diff()
    
    # 進行方向（ラジアン）
    df['direction'] = np.arctan2(dy, dx)
    
    # 最初の要素は前の要素と同じ方向とする
    df['direction'] = df['direction'].bfill()
    
    return df


def plot_direction_arrows(ax: Axes, df: pd.DataFrame, decimation_rate: int = 50, color: str = 'red') -> None:
    """進行方向の矢印を描画"""
    if 'direction' not in df.columns:
        df = calculate_direction(df)
    
    # 間引いてプロット
    decimated_df = df.iloc[::decimation_rate]
    
    # 矢印の長さ
    arrow_length = 1.5
    dx = arrow_length * np.cos(decimated_df['direction'])
    dy = arrow_length * np.sin(decimated_df['direction'])
    
    ax.quiver(decimated_df['x'], decimated_df['y'], dx, dy,
              angles='xy', scale_units='xy', scale=1,
              color=color, alpha=0.7, width=0.003)


def load_estimation_data(output_dir: str) -> Dict[str, pd.DataFrame]:
    """推定データを読み込む"""
    estimations = {}
    
    # positions.csvから推定軌跡を読み込む
    positions_file = os.path.join(output_dir, 'positions.csv')
    if os.path.exists(positions_file):
        df = pd.read_csv(positions_file)
        if {'timestamp', 'x', 'y', 'z'}.issubset(df.columns):
            estimations['overall'] = df
    
    # 各タグごとの推定データがある場合は読み込む（将来の拡張用）
    tag_files = glob.glob(os.path.join(output_dir, 'uwb_tag_*.csv'))
    for tag_file in tag_files:
        tag_id = os.path.basename(tag_file).replace('uwb_tag_', '').replace('.csv', '')
        df = pd.read_csv(tag_file)
        if {'timestamp', 'x', 'y', 'z'}.issubset(df.columns):
            estimations[f'tag_{tag_id}'] = df
    
    return estimations


def load_ground_truth(output_dir: str) -> Optional[pd.DataFrame]:
    """Ground truthデータを読み込む"""
    gt_file = os.path.join(output_dir, 'ground_truth.csv')
    if os.path.exists(gt_file):
        return pd.read_csv(gt_file)
    return None


def load_uwb_raw_data(output_dir: str) -> Dict[str, pd.DataFrame]:
    """UWBの生データを読み込む（LOS/NLOS情報含む）"""
    uwb_data = {}
    
    # UWBTデータ
    uwbt_file = os.path.join(output_dir, 'uwbt.csv')
    if os.path.exists(uwbt_file):
        df = pd.read_csv(uwbt_file)
        # タグごとに分離
        for tag_id in df['tag_id'].unique():
            uwb_data[f'uwbt_{tag_id}'] = df[df['tag_id'] == tag_id].copy()
    
    # UWBPデータ
    uwbp_file = os.path.join(output_dir, 'uwbp.csv')
    if os.path.exists(uwbp_file):
        df = pd.read_csv(uwbp_file)
        # タグごとに分離
        for tag_id in df['tag_id'].unique():
            uwb_data[f'uwbp_{tag_id}'] = df[df['tag_id'] == tag_id].copy()
    
    return uwb_data


def plot_overall_trajectory(estimations: Dict[str, pd.DataFrame], ground_truth: Optional[pd.DataFrame],
                          map_img: Optional[np.ndarray], map_origin: Tuple[float, float], 
                          map_ppm: float, arrow_decimation_rate: int, output_dir: str) -> None:
    """統合推定軌跡のプロット"""
    if 'overall' not in estimations:
        print("統合推定データが見つかりません")
        return
    
    df = estimations['overall']
    df = calculate_direction(df)
    
    _, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # マップを描画
    if map_img is not None:
        plot_map(ax, map_img, map_origin, map_ppm)
    
    # 推定軌跡をプロット
    scatter = ax.scatter(df['x'], df['y'], c=df['timestamp'], s=5, cmap='viridis', label='UWB推定軌跡')
    
    # 進行方向の矢印
    plot_direction_arrows(ax, df, decimation_rate=arrow_decimation_rate, color='red')
    
    # Ground truthがあれば描画
    if ground_truth is not None:
        ax.plot(ground_truth['x'], ground_truth['y'], 'k--', linewidth=2, alpha=0.5, label='Ground Truth')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('UWB統合推定軌跡')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.colorbar(scatter, ax=ax, label='Timestamp (s)')
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(output_dir, 'uwb_overall_trajectory.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.show()


def plot_los_nlos_analysis(uwb_raw_data: Dict[str, pd.DataFrame], 
                          output_dir: str) -> None:
    """各タグのLOS/NLOSプロット"""
    uwbt_tags = [key for key in uwb_raw_data.keys() if key.startswith('uwbt_')]
    
    for tag_key in uwbt_tags:
        tag_id = tag_key.replace('uwbt_', '')
        df = uwb_raw_data[tag_key]
        
        if 'nlos' not in df.columns:
            print(f"タグ {tag_id} にNLOS情報がありません")
            continue
        
        _, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # LOS/NLOSで色分けしてプロット
        los_mask = df['nlos'] <= 0.5
        nlos_mask = ~los_mask
        
        # 距離と角度情報がある場合のみプロット
        if {'distance', 'aoa_azimuth', 'aoa_elevation'}.issubset(df.columns):
            # タイムスタンプでカラーマップ
            if los_mask.any():
                ax.scatter(df.loc[los_mask, 'app_timestamp'], 
                          df.loc[los_mask, 'distance'],
                          c='green', s=10, alpha=0.6, label='LOS')
            
            if nlos_mask.any():
                ax.scatter(df.loc[nlos_mask, 'app_timestamp'], 
                          df.loc[nlos_mask, 'distance'],
                          c='red', s=10, alpha=0.6, label='NLOS')
            
            ax.set_xlabel('Timestamp (s)')
            ax.set_ylabel('Distance (m)')
            ax.set_title(f'タグ {tag_id} - LOS/NLOS 距離測定')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(output_dir, f'uwb_los_nlos_tag_{tag_id}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"保存: {save_path}")
        plt.show()


def plot_individual_tag_trajectories(output_dir: str, map_img: Optional[np.ndarray],
                                   map_origin: Tuple[float, float], map_ppm: float,
                                   arrow_decimation_rate: int) -> None:
    """タグごとの推定軌跡（個別）"""
    # GPOSデータから各タグの実際の位置を取得
    gpos_file = os.path.join(output_dir, 'gpos.csv')
    
    if not os.path.exists(gpos_file):
        print("GPOSデータが見つかりません")
        return
    
    gpos_df = pd.read_csv(gpos_file)
    
    # 各タグごとにデータを分離してプロット
    for tag_id in gpos_df['object_id'].unique():
        tag_data = gpos_df[gpos_df['object_id'] == tag_id].copy()
        
        # 個別プロット
        _, ax = plt.subplots(1, 1, figsize=(20, 10))
        
        # マップを描画
        if map_img is not None:
            plot_map(ax, map_img, map_origin, map_ppm)
        
        # タグの軌跡をプロット
        scatter = ax.scatter(tag_data['location_x'], tag_data['location_y'], 
                           c=tag_data['app_timestamp'], s=5, cmap='plasma',
                           label=f'タグ {tag_id} 軌跡')
        
        # 進行方向を計算してプロット
        tag_data_plot = tag_data.rename(columns={
            'location_x': 'x',
            'location_y': 'y',
            'location_z': 'z'
        })
        tag_data_plot = calculate_direction(tag_data_plot)
        plot_direction_arrows(ax, tag_data_plot, decimation_rate=arrow_decimation_rate, color='blue')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'タグ {tag_id} - 個別軌跡')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.colorbar(scatter, ax=ax, label='Timestamp (s)')
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(output_dir, f'uwb_trajectory_tag_{tag_id}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"保存: {save_path}")
        plt.show()


def print_statistics(uwb_raw_data: Dict[str, pd.DataFrame]) -> None:
    """統計情報の表示"""
    uwbt_tags = [key for key in uwb_raw_data.keys() if key.startswith('uwbt_')]
    
    print("\n=== LOS/NLOS統計 ===")
    for tag_key in uwbt_tags:
        tag_id = tag_key.replace('uwbt_', '')
        df = uwb_raw_data[tag_key]
        
        if 'nlos' in df.columns:
            los_count = (df['nlos'] <= 0.5).sum()
            nlos_count = (df['nlos'] > 0.5).sum()
            total = len(df)
            
            print(f"\nタグ {tag_id}:")
            print(f"  総測定数: {total}")
            print(f"  LOS: {los_count} ({los_count/total*100:.1f}%)")
            print(f"  NLOS: {nlos_count} ({nlos_count/total*100:.1f}%)")
            
            if 'distance' in df.columns:
                print(f"  平均距離: {df['distance'].mean():.2f}m")
                print(f"  最小距離: {df['distance'].min():.2f}m")
                print(f"  最大距離: {df['distance'].max():.2f}m")


def main() -> None:
    parser = argparse.ArgumentParser(description='UWB推定軌跡をプロット')
    parser.add_argument('--output-dir', default='../../output', help='出力ディレクトリ')
    parser.add_argument('--map-file', default='../../map/miraikan_5.bmp', help='マップファイル')
    parser.add_argument('--map-origin-x', type=float, default=-5.625, help='マップ原点X')
    parser.add_argument('--map-origin-y', type=float, default=-12.75, help='マップ原点Y')
    parser.add_argument('--map-ppm', type=int, default=100, help='pixels per meter')
    parser.add_argument('--arrow-decimation', type=int, default=50, help='矢印の間引き率')
    
    args = parser.parse_args()
    
    # 設定
    map_origin = (args.map_origin_x, args.map_origin_y)
    
    # マップ画像を読み込む
    try:
        map_img = load_map_image(args.map_file)
        print(f"マップ画像を読み込みました: {args.map_file}")
    except:
        print(f"警告: マップ画像が見つかりません: {args.map_file}")
        map_img = None
    
    # データを読み込む
    estimations = load_estimation_data(args.output_dir)
    print(f"\n推定データ: {list(estimations.keys())}")
    
    ground_truth = load_ground_truth(args.output_dir)
    if ground_truth is not None:
        print(f"\nGround truthデータを読み込みました: {len(ground_truth)}件")
    else:
        print("\nGround truthデータが見つかりません")
    
    uwb_raw_data = load_uwb_raw_data(args.output_dir)
    print(f"\nUWB生データ: {list(uwb_raw_data.keys())}")
    
    # プロット
    print("\n=== プロット開始 ===")
    
    # 1. 統合推定軌跡
    plot_overall_trajectory(estimations, ground_truth, map_img, map_origin, 
                          args.map_ppm, args.arrow_decimation, args.output_dir)
    
    # 2. LOS/NLOS分析
    plot_los_nlos_analysis(uwb_raw_data, args.output_dir)
    
    # 3. 個別タグ軌跡
    plot_individual_tag_trajectories(args.output_dir, map_img, map_origin, 
                                   args.map_ppm, args.arrow_decimation)
    
    # 4. 統計情報
    print_statistics(uwb_raw_data)
    
    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()