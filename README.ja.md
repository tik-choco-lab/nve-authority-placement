# NVE Authority Placement

[English](README.md) | [日本語](README.ja.md)

P2P型のNetworked Virtual Environment（複数人が参加する仮想空間）で、Shared Entity（NPCなど共有対象）のAuthority（正式な状態を決めるPeer）をどこに置くべきかを比較するための実験基盤です。

実際のプレイヤーログではなく、乱数とパラメータから作った合成ワークロードです。

## Quick start

前提: Python 3.11以上、[uv](https://docs.astral.sh/uv/)。

以下はリポジトリのルートで、Bashなどのglob展開に対応したシェルから実行します。

```bash
uv sync --locked

# 1. データセットを生成する（10 Peers・3 Entities・60秒・4シナリオ×seed 1,2,3 = 12件）
uv run nve-dataset generate-suite --config config.yaml

# 2. 検証・再現確認
uv run nve-dataset validate datasets/small/concentrated/seed_001
uv run nve-dataset reproduce-check datasets/small/concentrated/seed_001

# 3. 12件を1枚の表にまとめる
uv run nve-dataset aggregate datasets/small/*/seed_* --output datasets/small/summary_all.csv

# 4. 配置方式を実行して比較する
uv run nve-policy replay datasets/small/*/seed_* --config policies.yaml
uv run nve-policy compare datasets/small/*/seed_* --output datasets/small/policy_comparison.csv

# 5. テスト
uv run pytest
```

このデモ用データセットは、論文の実験で使う構成より小規模です。

## ドキュメント一覧

| ドキュメント | 内容 |
|---|---|
| [REPRODUCING.md](REPRODUCING.md) | 実験コマンド、seed、統計処理、出力ファイルの対応 |
| [docs/DATASET_DESIGN.md](docs/DATASET_DESIGN.md) | シナリオ、生成規則、仮定、検証方法 |
| [docs/CSV_SCHEMA.md](docs/CSV_SCHEMA.md) | 各CSVの列定義 |
| [docs/POLICY_EVALUATION.md](docs/POLICY_EVALUATION.md) | replayの処理順、配置方式、コストモデル、制約、出力形式 |

## ライセンス

コードは [MIT License](LICENSE)、生成データは [CC BY 4.0](datasets/LICENSE) です。
