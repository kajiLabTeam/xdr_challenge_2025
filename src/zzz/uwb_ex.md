# UWB位置推定システムの説明

## 概要
本システムは、Ultra-Wideband (UWB) 技術を使用した屋内位置推定システムです。複数のUWBタグからのデータを収集し、各タグごとに独立した推定軌跡を生成します。

## システム構成

### 1. データ収集
システムは2種類のUWBデータを処理します：

- **UWBT (UWB with Time)**: 方位角・仰角を含む測定データ
  - 距離 (distance)
  - 方位角 (aoa_azimuth)
  - 仰角 (aoa_elevation)  
  - NLOS状態 (nlos: 0.0=LOS, 1.0=NLOS)

- **UWBP (UWB with Position)**: 方向ベクトルを含む測定データ
  - 距離 (distance)
  - 方向ベクトル (direction_vec_x, y, z)

### 2. LOS/NLOS判定

#### Line of Sight (LOS) vs Non-Line of Sight (NLOS)
- **LOS**: タグとアンカー間に障害物がない直接見通し状態
- **NLOS**: 障害物により電波が反射・回折する状態

#### 判定基準
```python
nlos_threshold = 0.5  # 閾値
is_los = nlos_value < nlos_threshold
```
- nlos値が0.0の場合：完全なLOS状態
- nlos値が1.0の場合：完全なNLOS状態

### 3. 位置推定アルゴリズム

#### 3.1 座標変換
UWBTデータの球面座標をデカルト座標に変換：

```python
# 球面座標からローカルデカルト座標への変換
x = distance * cos(elevation) * sin(azimuth)
y = distance * cos(elevation) * cos(azimuth)
z = distance * sin(elevation)
```

#### 3.2 座標系変換
タグの姿勢（クォータニオン）を使用してワールド座標系に変換：

```python
# タグの姿勢で回転してワールド座標に変換
world_point = R_tag.apply(local_point) + tag_location
```

#### 3.3 信頼度計算
各測定の信頼度を計算：

```python
# 基本信頼度（距離が遠いほど低下）
confidence = 1.0 / (1.0 + distance * 0.1)

# NLOS補正
if is_los:
    confidence *= 1.0  # LOSの場合は維持
else:
    confidence *= 0.5  # NLOSの場合は半減
```

### 4. タグ別軌跡管理

#### 4.1 独立軌跡生成
各タグ（例：3583WAA、3636DWF、3637RLJ）ごとに独立した推定軌跡を生成し、以下の情報を保持：

- 位置座標 (x, y, z)
- LOS/NLOS状態
- 信頼度
- 測定方法 (UWBT/UWBP)

#### 4.2 最終位置決定
複数タグの中から最も信頼度の高いタグの位置を最終推定位置として採用。

### 5. データ出力

#### 5.1 CSVファイル
各タグごとに個別のCSVファイルを生成：

```csv
x,y,z,is_los,confidence,method
41.082,-9.608,1.069,True,1.897,UWBP
41.222,-9.606,1.232,True,2.844,UWBP
41.027,-9.640,1.000,False,0.949,UWBT
```

#### 5.2 可視化
各タグの軌跡を個別の画像ファイルとして出力：

- **青色の点/線**: LOS状態の測定
- **赤色の点/線**: NLOS状態の測定
- **緑の四角**: 開始位置
- **オレンジの三角**: 終了位置

統計情報の表示：
- 総測定点数
- LOS/NLOS比率
- UWBT/UWBP測定数
- 総移動距離

## 実装ファイル構成

### コアモジュール
- `src/lib/localizer/uwb.py`: UWB位置推定の主要ロジック
- `src/lib/recorder/uwbt.py`: UWBTデータの記録
- `src/lib/recorder/uwbp.py`: UWBPデータの記録
- `src/lib/visualizer/_visualizer.py`: 軌跡の可視化

### データ構造
```python
class TagEstimate:
    tag_id: str          # タグID
    position: ndarray    # 推定位置
    confidence: float    # 信頼度
    distance: float      # 測定距離
    method: str          # 'UWBT' or 'UWBP'
    is_los: bool        # LOS/NLOS状態

class PositionWithLOS:
    x: float            # X座標
    y: float            # Y座標
    z: float            # Z座標
    is_los: bool        # LOS/NLOS状態
    confidence: float   # 信頼度
    method: str         # 測定方法
```

## 使用方法

### 基本実行
```bash
# デモモード（ローカルデータ使用）
python main.py --demo

# 即時実行モード（リアルタイム処理なし）
python main.py --demo --immediate

# プロット表示なし
python main.py --demo --immediate --no-save-plot-map
```

### 出力ファイル
実行後、`output/`ディレクトリに以下のファイルが生成されます：

- `{trial_id}_{timestamp}_est.csv`: 最終推定結果
- `{trial_id}_{timestamp}_tag_{tag_id}.csv`: 各タグの軌跡データ（LOS情報付き）
- `{trial_id}_{timestamp}_map.png`: 最終推定結果のマップ
- `{trial_id}_{timestamp}_map_tag_{tag_id}.png`: 各タグの軌跡マップ（LOS/NLOS可視化）

## パフォーマンス考慮事項

### メモリ管理
- 各タグの履歴は最大10個まで保持（`max_history = 10`）
- 古いデータは自動的に削除

### 処理効率
- 最新20個のデータのみを処理対象とする
- NLOSデータも活用することで、データの有効利用率を向上

### 精度向上のポイント
1. **LOS優先**: LOS状態の測定により高い信頼度を付与
2. **距離補正**: 距離が遠いほど信頼度を低減
3. **複数タグ活用**: 最も信頼度の高いタグを動的に選択
4. **方法別重み付け**: UWBTの測定に1.2倍の重みを付与（より高精度なため）

## トラブルシューティング

### 問題: NLOSデータが多い場合
- 原因: 環境に障害物が多い
- 対策: `nlos_threshold`を調整（デフォルト: 0.5）

### 問題: 軌跡が不連続
- 原因: タグ間の切り替えが頻繁
- 対策: 信頼度の重み付けパラメータを調整

### 問題: 推定精度が低い
- 原因: 単一タグのデータ不足
- 対策: 複数タグの配置を最適化

## 今後の改良予定

1. **カルマンフィルタの導入**: 軌跡の平滑化
2. **機械学習ベースのNLOS検出**: より高精度なNLOS判定
3. **リアルタイム可視化**: 推定過程の動的表示
4. **センサフュージョン**: IMUデータとの統合