# Texas Hold'em Poker Game Implementation in Python

PythonによるテキサスホールデムポーカーのCLI実装。ルールベースの対戦AIに加え、**GTO（Game Theory Optimal）CPU** を搭載。Regret Matching による混合戦略と事前学習・事後学習に対応している。

## 起動方法

```bash
uv run python main.py
```

キー操作: `c` = check/call、`f` = fold、`r`/`b` = raise/bet

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
| `gto_selfplay.py` | 事前学習 CLI スクリプト |
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