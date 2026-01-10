# 4×4×4 Cube Analysis Simulator

4×4×4ルービックキューブのシミュレーター／解析ツールです。エッジペア/キューブレット検査などの解析機能も含みます。

## 背景

本プロジェクトは、伊那北高校の数学一班での理数探求の一環として作成したものです。

## 注意事項および免責

- 設計・実装は素人による個人の開発で、GitHub Copilot を使用してコードを書いています。
- バグ、不正確な挙動、仕様の不整合が残っている可能性があります。
- 本リポジトリの内容は無保証で提供されます。利用・改変・実行は自己責任でお願いします。
- 高い信頼性・安全性が求められる用途では使用しないでください。

## 主な機能

- **Web GUI**: ブラウザ上で操作・状態確認（WebSocket対応）
- **解析**: エッジ追跡、位置解析、バッチ解析、Excel出力、キューブレット検査

## セットアップ

### 必要要件

- Python **3.10+**

### インストール

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## クイックスタート

### 1) Web GUIを使用

```bash
python run_web_gui.py
```

- ブラウザで `http://localhost:8888` を開きます
- Flaskの `SECRET_KEY` は環境変数 `SECRET_KEY` で上書き（未設定でも起動します）

macOS/Linux:

```bash
export SECRET_KEY="change-me"
python run_web_gui.py
```

Windows (cmd):

```bat
set SECRET_KEY=change-me
python run_web_gui.py
```

### 2) CLIを使用(非推奨)

```bash
python -m src.cli -i
```

- `show` で3D表示、`text` で展開図表示、`inspect` で検査メニュー

#### Inspector（検査機能）について

- CLIの `inspect` でキューブレット情報とエッジペア検査ができます
- Web GUI では以下のAPIが使えます
	- `GET /api/inspector/cubelets`
	- `GET /api/inspector/edge_pairs`
	- `GET /api/inspector/edge_pair/<pair_id>`

## ドキュメント

- 回転行列仕様: [ROTATION_MATRIX_SPEC.md](ROTATION_MATRIX_SPEC.md)

## ライセンス

[LICENSE](LICENSE) を参照してください（MITライセンスベース + **クレジット表記要件**あり）。
