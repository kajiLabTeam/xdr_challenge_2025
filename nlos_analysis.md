# NLOSデータが少ない理由の分析

## 現在の処理フローと問題点

### 1. データ処理の流れ

```
元データ (1.txt)
    ↓
UWBTDataRecorder / UWBPDataRecorder
    ↓
estimate_from_uwbt() / estimate_from_uwbp()
    ↓ [最新10個のデータから推定]
estimate_position_per_tag()
    ↓ [重み付き平均で統合]
estimate_uwb()
    ↓
軌跡データ (tag_trajectories_with_los)
```

### 2. NLOSデータが失われる主な原因

#### 原因1: 重み付き平均による情報の喪失

`estimate_position_per_tag()`関数では：

```python
# 全推定の重み付き平均を計算
weighted_position = np.sum(positions * weights.reshape(-1, 1), axis=0) / total_weight

# しかし、LOS/NLOS情報は「最も信頼度の高い」推定から取得
best_estimate = max(all_estimates, key=lambda e: e.confidence)
```

**問題点**：
- 位置は全推定の平均
- LOS/NLOS情報は最も信頼度の高い1つの推定のみから取得
- NLOSの信頼度は50%に低減されるため、ほぼ常にLOSが選ばれる

#### 原因2: 信頼度の計算方法

```python
# 基本信頼度（距離に基づく）
confidence = 1.0 / (1.0 + distance * 0.1)

# NLOSの場合
if not is_los:
    confidence *= 0.5  # 半減
```

**具体例**：
- 距離0.5mのLOS: 信頼度 = 1.0 / 1.05 = 0.952
- 距離0.5mのNLOS: 信頼度 = 0.952 * 0.5 = 0.476
- 距離5.0mのLOS: 信頼度 = 1.0 / 1.5 = 0.667

→ 遠距離のLOSでも、近距離のNLOSより高い信頼度

#### 原因3: UWBPデータの扱い

```python
# UWBPデータは常にLOSとして扱われる
is_los=True  # UWBPはNLOS情報なし
```

UWBPデータには元々NLOS情報がないため、全てLOSとして記録される。

### 3. 実際のデータ分布

#### 元データ（1.txt）のNLOS分布：
- UWBT全体: 765/3939 (19.4%) がNLOS
- Tag 3583WAA: 548/1733 (31.6%)
- Tag 3636DWF: 59/1113 (5.3%)
- Tag 3637RLJ: 158/1093 (14.5%)

#### 最終軌跡のNLOS分布：
- Tag 3583WAA: 16/1650 (0.9%)
- Tag 3636DWF: 26/1141 (2.2%)
- Tag 3637RLJ: 50/913 (5.4%)

**大幅な減少の理由**：
1. NLOSデータの信頼度が低いため、「最も信頼度の高い推定」として選ばれない
2. UWBPデータ（全てLOS扱い）との統合により希釈される
3. 時間窓（最新10個）の制限により、古いNLOSデータが除外される

### 4. 改善案

#### 案1: 全推定データを軌跡として保存
重み付き平均を使わず、各推定を独立して軌跡に追加

#### 案2: NLOSペナルティの調整
信頼度の低減率を0.5から0.8程度に緩和

#### 案3: 推定方法の分離
UWBT（NLOS情報あり）とUWBP（NLOS情報なし）を別々に処理

#### 案4: 統計的アプローチ
一定時間窓内のNLOS比率を計算し、その情報を保持

### 5. 現在の処理による影響

**メリット**：
- ノイズの少ない安定した軌跡
- 高精度な位置推定

**デメリット**：
- NLOS環境の実態が反映されない
- 環境分析や改善のための情報が失われる
- 実際のUWB通信品質が評価できない