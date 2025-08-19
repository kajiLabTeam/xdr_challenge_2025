#!/usr/bin/env python
"""
NLOS分布分析スクリプト
デモデータのLOS/NLOS分布を詳細に分析・可視化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_demo_data(file_path='src/api/trials/1.txt'):
    """デモデータを読み込んでDataFrameに変換"""
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) > 0 and parts[0] in ['UWBT', 'UWBP']:
                sensor_type = parts[0]
                if sensor_type == 'UWBT' and len(parts) == 8:
                    data.append({
                        'type': sensor_type,
                        'app_time': float(parts[1]),
                        'sensor_time': float(parts[2]),
                        'tag_id': parts[3],
                        'distance': float(parts[4]),
                        'aoa_azimuth': float(parts[5]),
                        'aoa_elevation': float(parts[6]),
                        'nlos': float(parts[7])
                    })
                elif sensor_type == 'UWBP' and len(parts) == 8:
                    data.append({
                        'type': sensor_type,
                        'app_time': float(parts[1]),
                        'sensor_time': float(parts[2]),
                        'tag_id': parts[3],
                        'distance': float(parts[4]),
                        'dir_x': float(parts[5]),
                        'dir_y': float(parts[6]),
                        'dir_z': float(parts[7]),
                        'nlos': None  # UWBPにはNLOS情報なし
                    })
    
    return pd.DataFrame(data)

def analyze_nlos_distribution(df):
    """NLOS分布を分析"""
    print("="*60)
    print("NLOS分布分析")
    print("="*60)
    
    # UWBT データのみを分析（NLOS情報があるため）
    uwbt_df = df[df['type'] == 'UWBT'].copy()
    
    # 全体の統計
    total_uwbt = len(uwbt_df)
    nlos_count = (uwbt_df['nlos'] == 1.0).sum()
    los_count = (uwbt_df['nlos'] == 0.0).sum()
    
    print(f"\n全体統計:")
    print(f"  総UWBT測定数: {total_uwbt}")
    print(f"  LOS (nlos=0.0): {los_count} ({los_count/total_uwbt*100:.1f}%)")
    print(f"  NLOS (nlos=1.0): {nlos_count} ({nlos_count/total_uwbt*100:.1f}%)")
    
    # タグごとの統計
    print(f"\nタグ別統計:")
    tags = uwbt_df['tag_id'].unique()
    tag_stats = {}
    
    for tag in sorted(tags):
        tag_data = uwbt_df[uwbt_df['tag_id'] == tag]
        tag_total = len(tag_data)
        tag_nlos = (tag_data['nlos'] == 1.0).sum()
        tag_los = (tag_data['nlos'] == 0.0).sum()
        tag_stats[tag] = {
            'total': tag_total,
            'los': tag_los,
            'nlos': tag_nlos,
            'nlos_ratio': tag_nlos/tag_total*100
        }
        print(f"  {tag}:")
        print(f"    総数: {tag_total}")
        print(f"    LOS: {tag_los} ({tag_los/tag_total*100:.1f}%)")
        print(f"    NLOS: {tag_nlos} ({tag_nlos/tag_total*100:.1f}%)")
    
    return uwbt_df, tag_stats

def plot_nlos_timeline(uwbt_df, output_dir='output'):
    """時系列でLOS/NLOSをプロット"""
    Path(output_dir).mkdir(exist_ok=True)
    
    tags = sorted(uwbt_df['tag_id'].unique())
    
    # 全タグの時系列プロット（1つの図）
    fig, axes = plt.subplots(len(tags), 1, figsize=(15, 4*len(tags)), sharex=True)
    if len(tags) == 1:
        axes = [axes]
    
    for idx, tag in enumerate(tags):
        tag_data = uwbt_df[uwbt_df['tag_id'] == tag].copy()
        tag_data = tag_data.sort_values('app_time')
        
        ax = axes[idx]
        
        # LOS/NLOSを色分けしてプロット
        los_data = tag_data[tag_data['nlos'] == 0.0]
        nlos_data = tag_data[tag_data['nlos'] == 1.0]
        
        # 背景に帯を描画（NLOS期間を強調）
        for i in range(len(tag_data) - 1):
            if tag_data.iloc[i]['nlos'] == 1.0:
                ax.axvspan(tag_data.iloc[i]['app_time'], 
                          tag_data.iloc[i+1]['app_time'],
                          alpha=0.2, color='red', label='NLOS period' if i == 0 else '')
        
        # 距離をプロット
        ax.scatter(los_data['app_time'], los_data['distance'], 
                  c='blue', s=10, alpha=0.6, label='LOS')
        ax.scatter(nlos_data['app_time'], nlos_data['distance'], 
                  c='red', s=20, alpha=0.8, marker='x', label='NLOS')
        
        ax.set_ylabel('Distance (m)')
        ax.set_title(f'Tag {tag} - Distance over Time (LOS/NLOS)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # NLOS比率を右側に表示
        nlos_ratio = len(nlos_data) / len(tag_data) * 100
        ax.text(1.02, 0.5, f'NLOS:\n{nlos_ratio:.1f}%', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='center')
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('UWBT Distance Measurements - LOS vs NLOS Timeline', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'nlos_timeline.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n時系列プロットを保存: {output_file}")
    plt.show()

def plot_nlos_statistics(tag_stats, output_dir='output'):
    """統計情報の可視化"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. タグ別NLOS比率の棒グラフ
    ax = axes[0, 0]
    tags = list(tag_stats.keys())
    nlos_ratios = [tag_stats[tag]['nlos_ratio'] for tag in tags]
    colors = ['red' if ratio > 20 else 'orange' if ratio > 10 else 'blue' for ratio in nlos_ratios]
    
    bars = ax.bar(tags, nlos_ratios, color=colors, alpha=0.7)
    ax.set_ylabel('NLOS Ratio (%)')
    ax.set_title('NLOS Ratio by Tag')
    ax.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20% threshold')
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
    
    # 値を棒の上に表示
    for bar, ratio in zip(bars, nlos_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}%', ha='center', va='bottom')
    ax.legend()
    
    # 2. 測定数の比較（積み上げ棒グラフ）
    ax = axes[0, 1]
    los_counts = [tag_stats[tag]['los'] for tag in tags]
    nlos_counts = [tag_stats[tag]['nlos'] for tag in tags]
    
    x = np.arange(len(tags))
    width = 0.6
    
    ax.bar(x, los_counts, width, label='LOS', color='blue', alpha=0.7)
    ax.bar(x, nlos_counts, width, bottom=los_counts, label='NLOS', color='red', alpha=0.7)
    
    ax.set_ylabel('Number of Measurements')
    ax.set_title('Measurement Count by Tag (LOS vs NLOS)')
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.legend()
    
    # 合計値を表示
    for i, (los, nlos) in enumerate(zip(los_counts, nlos_counts)):
        ax.text(i, los + nlos, f'{los + nlos}', ha='center', va='bottom')
    
    # 3. パイチャート（全体のLOS/NLOS比率）
    ax = axes[1, 0]
    total_los = sum(los_counts)
    total_nlos = sum(nlos_counts)
    
    wedges, texts, autotexts = ax.pie([total_los, total_nlos], 
                                       labels=['LOS', 'NLOS'],
                                       colors=['blue', 'red'],
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       explode=(0, 0.1))
    
    ax.set_title(f'Overall LOS/NLOS Distribution\n(Total: {total_los + total_nlos} measurements)')
    
    # 4. 距離とNLOSの関係（箱ひげ図）
    ax = axes[1, 1]
    
    # 全データを読み込み
    df = load_demo_data()
    uwbt_df = df[df['type'] == 'UWBT'].copy()
    
    los_distances = uwbt_df[uwbt_df['nlos'] == 0.0]['distance']
    nlos_distances = uwbt_df[uwbt_df['nlos'] == 1.0]['distance']
    
    bp = ax.boxplot([los_distances, nlos_distances], 
                     labels=['LOS', 'NLOS'],
                     patch_artist=True)
    
    colors = ['blue', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance Distribution: LOS vs NLOS')
    ax.grid(True, alpha=0.3)
    
    # 統計値を追加
    stats_text = f"LOS: mean={los_distances.mean():.2f}m, std={los_distances.std():.2f}m\n"
    stats_text += f"NLOS: mean={nlos_distances.mean():.2f}m, std={nlos_distances.std():.2f}m"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('NLOS Statistical Analysis', fontsize=14)
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'nlos_statistics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"統計プロットを保存: {output_file}")
    plt.show()

def main():
    """メイン処理"""
    print("NLOS分析を開始します...")
    
    # データ読み込み
    df = load_demo_data()
    print(f"データ読み込み完了: {len(df)} 件")
    
    # NLOS分布分析
    uwbt_df, tag_stats = analyze_nlos_distribution(df)
    
    # 可視化
    plot_nlos_timeline(uwbt_df)
    plot_nlos_statistics(tag_stats)
    
    print("\n分析完了！")

if __name__ == '__main__':
    main()