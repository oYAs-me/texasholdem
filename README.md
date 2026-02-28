# Texas Hold'em Poker Game Implementation in Python

PythonによるテキサスホールデムポーカーのCLI実装。ルールベースの対戦AIに加え、**GTO（Game Theory Optimal）CPU** を搭載。Regret Matching による混合戦略と事前学習・事後学習に対応している。

## 起動方法

```bash
# 基本（人間1人 vs CPU 5人）
uv run python main.py

# CPUのみで10回対戦させ、結果の統計を表示（詳細ログは抑制）
uv run python main.py --cpu-only -n 10 -s
```

キー操作: `c` = check/call、`f` = fold、`r`/`b` = raise/bet

## 実行オプション (main.py)

`main.py` では、人間対CPUの対戦だけでなく、CPUのみのシミュレーションや複数回マッチの実行が可能です。

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-n`, `--num-matches` | 1 | 実行するマッチ数（誰か一人が勝ち残るまでを1マッチとする） |
| `--cpu-only` | — | 人間プレイヤーを除外し、CPUのみで対戦させる |
| `-s`, `--silent` | — | ラウンドごとの詳細なログ出力を抑制し、マッチ終了通知と最終結果のみ表示する |
| `--chips` | 1000 | 各プレイヤーの初期チップ数 |

## ファイル構成

| ファイル | 役割 |
|---------|------|
| `card.py` | `Card` / `Hand` / `Board` クラス、デッキ生成 |
| `hand_strength.py` | 役判定（`evaluate_hand`）、`EvaluatedHand` |
| `probability.py` | エクイティ計算（Monte Carlo）、ハンド分布計算 |
| `player.py` | `Player` 基底クラス、`HumanPlayer`、ルールベース CPU 3種 |
| `game.py` | ゲーム進行、ベッティングラウンド、ショーダウン |
| `learning_game.py` | `Game` サブクラス。ラウンド後に `on_round_end` フックを呼ぶ |
| `gto_strategy.py` | ゲーム状態の抽象化、ヒューリスティック初期戦略テーブル |
| `gto_cfr.py` | Regret Matching エンジン（SimpleMCCFR）、JSON 永続化 |
| `gto_cpu.py` | `GtoCpu` クラス。混合戦略による意思決定、GTO ベットサイジング |
| `gto_selfplay.py` | 事前学習 CLI。完了後に自動でレア学習も実行 |
| `gto_rare_training.py` | 珍しい状況（visit_count 不足）に特化した集中学習 CLI |
| `precompute_preflop.py` | プリフロップ分布の事前計算（`preflop_distributions.json` を生成） |

## テスト

```bash
uv run python -m unittest test_game.py -v
```

## CPU の種類

| クラス | 特徴 |
|-------|------|
| `ConservativeCpu` | エクイティが高い時だけベット。保守的 |
| `BalancedCpu` | エクイティとポットオッズのバランスで判断 |
| `AggressiveCpu` | 一定確率でブラフ込みのアグレッシブな行動 |
| `GtoCpu` | Regret Matching による混合戦略。プレイを重ねるほど洗練される |

## GTO CPU の事前学習

ゲーム開始前に自己対戦で戦略を学習させる。学習結果は `gto_strategy.json` に保存され、次回起動時に引き継がれる。`main.py` でのプレイ中も自動的に事後学習が行われ、蓄積され続ける。

selfplay 正常終了後は自動的にレア状態の集中学習（`gto_rare_training`）も実行される。`--no-rare` でスキップ可能。

```bash
uv run python gto_selfplay.py

# オプション一覧
uv run python gto_selfplay.py --help
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--hands N` | 1000 | 学習ハンド数（`--minutes` と同時指定不可） |
| `--minutes M` | — | 学習時間（分）。終了時に自動保存 |
| `--players N` | 4 | テーブル人数（2〜6） |
| `--workers N` | CPU コア数 | 並列ワーカー数 |
| `--sims N` | 200 | Monte Carlo 試行数（少ないほど高速・精度低） |
| `--save PATH` | `gto_strategy.json` | 保存先 |
| `--no-rare` | — | selfplay 後のレア学習をスキップ |

## レア状態の集中学習

`visit_count` が少ない状態（珍しいボード・状況）に絞って学習を強化する。**時間指定のみ**（ハンド数指定なし）。

```bash
uv run python gto_rare_training.py

# オプション一覧
uv run python gto_rare_training.py --help
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--minutes M` | 5 | 学習時間（分） |
| `--threshold N` | 100 | visit_count がこの値未満をレアとみなす |
| `--players N` | 4 | テーブル人数（2〜6） |
| `--workers N` | CPU コア数 | 並列ワーカー数 |
| `--sims N` | 200 | Monte Carlo 試行数 |
| `--save PATH` | `gto_strategy.json` | 保存先 |