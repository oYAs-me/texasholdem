"""
gto_selfplay.py

GtoCpu 同士の自動対戦による事前学習スクリプト。
ゲーム出力を抑制しながら高速に N ハンドをシミュレートし、
学習済み戦略を JSON に保存する。

使用例:
    uv run python gto_selfplay.py                         # 1000 ハンド / 4 人
    uv run python gto_selfplay.py --hands 5000
    uv run python gto_selfplay.py --hands 10000 --players 6
    uv run python gto_selfplay.py --save custom.json --sims 200
    uv run python gto_selfplay.py --verbose               # ゲーム出力を表示
"""
from __future__ import annotations

import argparse
import contextlib
import io

from tqdm import tqdm

from gto_cpu import GtoCpu
from learning_game import LearningGame


def run_selfplay(
    num_hands: int,
    num_players: int,
    save_path: str,
    num_simulations: int,
    verbose: bool,
) -> None:
    start_chips = 1000

    players = [
        GtoCpu(
            f"GTO-{i}",
            chips=start_chips,
            save_path=save_path,
            num_simulations=num_simulations,
        )
        for i in range(num_players)
    ]

    game = LearningGame(players, start_chips=start_chips, sb=10, bb=20)

    print(
        f"事前学習開始: {num_players} 人 × {num_hands} ハンド"
        f" | エクイティ計算: {num_simulations} 試行 → {save_path}"
    )

    pbar = tqdm(range(num_hands), desc="Self-Play", unit="hands")
    save_interval = max(num_hands // 20, 50)  # 全体の 5% ごとに保存

    for hand_idx in pbar:
        # バスト時のリバイ（チップを全員リセット）
        active = [p for p in game.players if p.chips > 0]
        if len(active) < 2:
            for p in game.players:
                p.chips = start_chips
                p.status = "active"

        # ゲーム出力を抑制しながら 1 ラウンド実行
        if verbose:
            game.play_round()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                game.play_round()

        # 定期保存
        if (hand_idx + 1) % save_interval == 0:
            players[0].cfr.save(save_path)
            num_states = len(players[0].cfr.visit_count)
            pbar.set_postfix(states=num_states, saved=hand_idx + 1)

    # 最終保存
    players[0].cfr.save(save_path)
    num_states = len(players[0].cfr.visit_count)
    print(f"\n✓ 学習完了: {num_states} 状態を学習 → {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GTO CPU 事前学習スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hands", type=int, default=1000, metavar="N",
        help="学習ハンド数",
    )
    parser.add_argument(
        "--players", type=int, default=4, choices=range(2, 7), metavar="N",
        help="テーブル人数 (2〜6)",
    )
    parser.add_argument(
        "--save", type=str, default="gto_strategy.json", metavar="PATH",
        help="戦略 JSON の保存先",
    )
    parser.add_argument(
        "--sims", type=int, default=200, metavar="N",
        help="エクイティ計算の Monte Carlo 試行数（少ないほど高速・精度低）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="ゲームの出力を表示する（デバッグ用）",
    )
    args = parser.parse_args()

    run_selfplay(
        num_hands=args.hands,
        num_players=args.players,
        save_path=args.save,
        num_simulations=args.sims,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
