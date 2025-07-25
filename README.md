# xDR Challenge 2025

## セットアップ
### 仮想環境の作成
```bash
python -m .venv venv
```

### 仮想環境のアクティベート
VSCode の `Python: Select Interpreter` で `.venv` を選択するか、以下のコマンドを実行します。

```bash
source /Users/satooru/Documents/kajilab/xdr-challenge/xdr_challenge_2025/.venv/bin/activate
```

### 必要なパッケージのインストール
```bash
pip install -r requirements.txt
```

### 環境変数の設定
※本番環境のみ

```bash
cp .env.example .env.competition
```


## 実行方法
### デモ環境
```bash
python main.py --demo
```

### 本番環境
```bash
python main.py
```
