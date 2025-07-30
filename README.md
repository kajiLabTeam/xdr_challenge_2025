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

### オプション
| オプション     | 省略形 | デフォルト値 | 説明                                                                               |
| -------------- | ------ | ------------ | ---------------------------------------------------------------------------------- |
| `--demo`       | `-d`   | `False`      | デモ環境で実行します。環境変数は `.env.demo` を使用します。                        |
| `--maxwait`    | `-w`   | `0.5`        | nextdata 取得・送信時の間隔。単位は秒                                              |
| `--loglevel`   | `-l`   | `info`       | ログの出力レベル。`debug`, `info`, `warning`, `error`, `critical` から選択します。 |
| `--run-server` | `-r`   | `False`      | ローカルの EvAAL API サーバーを起動します。本番環境では使用できません。            |
| `--output-dir` | `-o`   | `output`     | 出力ディレクトリ。                                                                 |

### 例
```bash
python main.py -w 0.5 -l debug -r --demo
```
