"""
gto_selfplay.py

GtoCpu 同士の自動対戦による事前学習スクリプト。
マルチプロセスで複数テーブルを並列実行し、結果をマージして保存する。
正常終了後、自動的にレア状態の集中学習（gto_rare_training）も実行する。

使用例:
    uv run python gto_selfplay.py                          # 1000 ハンド / 自動並列
    uv run python gto_selfplay.py --minutes 30             # 30 分間学習
    uv run python gto_selfplay.py --hands 5000
    uv run python gto_selfplay.py --hands 10000 --players 6
    uv run python gto_selfplay.py --workers 1              # シングルプロセス
    uv run python gto_selfplay.py --save custom.json --sims 200
    uv run python gto_selfplay.py --verbose                # ゲーム出力を表示
    uv run python gto_selfplay.py --no-rare                # レア学習をスキップ
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from gto_cpu import GtoCpu
from gto_cfr import SimpleMCCFR
from gto_cfr_utils import merge_cfr_data, apply_merged_data, load_base, save_merged, cfr_to_dict
from gto_rare_training import run_rare_training
from learning_game import LearningGame


# ──────────────────────────────────────────────
# ワーカー関数（トップレベル必須: Windows の spawn で pickle 可能にするため）
# ──────────────────────────────────────────────

def _run_chunk(args: tuple) -> dict:
    """
    サブプロセスで実行される学習チャンク。
    独立した GtoCpu × N でゲームを回し、学習済み CFR データを返す。
    ディスクへの保存は行わない（os.devnull に捨てる）。
    """
    chunk_hands, num_players, start_chips, num_simulations = args

    players = [
        GtoCpu(
            f"GTO-{i}",
            chips=start_chips,
            save_path=os.devnull,       # ワーカー内では保存しない
            num_simulations=num_simulations,
        )
        for i in range(num_players)
    ]
    game = LearningGame(players, start_chips=start_chips, sb=10, bb=20)

    for _ in range(chunk_hands):
        active = [p for p in game.players if p.chips > 0]
        if len(active) < 2:
            for p in game.players:
                p.chips = start_chips
                p.status = "active"
        with contextlib.redirect_stdout(io.StringIO()):
            game.play_round()

    # 全プレイヤーの CFR データをマージして返す（対称的な自己対戦なので合算可能）
    return merge_cfr_data([cfr_to_dict(p.cfr) for p in players])

# ──────────────────────────────────────────────
# メイン学習ループ
# ──────────────────────────────────────────────

# 時間ベースモードでのバッチサイズ（ワーカー1つあたりのハンド数）
_BATCH_HANDS_PER_WORKER = 50


def run_selfplay(
    num_hands: int | None,
    num_players: int,
    save_path: str,
    num_simulations: int,
    num_workers: int,
    verbose: bool,
    max_seconds: float | None = None,
    run_rare: bool = True,
) -> None:
    start_chips = 1000
    effective_workers = num_workers if num_hands is None else min(num_workers, num_hands)
    timed = max_seconds is not None

    mode_str = f"{int(max_seconds // 60)}分間" if timed else f"{num_hands} ハンド"
    print(
        f"事前学習開始: {num_players} 人 × {mode_str}"
        f" | workers={effective_workers}"
        f" | sims={num_simulations}"
        f" → {save_path}"
    )

    base_data = load_base(save_path)
    all_results: list[dict] = [base_data]
    chunk_args = (_BATCH_HANDS_PER_WORKER, num_players, start_chips, num_simulations)

    # ── 時間ベースモード ────────────────────────────
    if timed:
        end_time = time.monotonic() + max_seconds
        total_hands = 0

        pbar = tqdm(
            desc="Self-Play",
            unit="hands",
            bar_format="{desc}: {n_fmt} hands [{elapsed} elapsed, {rate_fmt}]{postfix}",
        )
        try:
            while time.monotonic() < end_time:
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    break

                if effective_workers == 1:
                    result = _run_chunk(chunk_args)
                    all_results.append(result)
                    total_hands += _BATCH_HANDS_PER_WORKER
                    pbar.update(_BATCH_HANDS_PER_WORKER)
                else:
                    batch = [chunk_args] * effective_workers
                    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                        futures = [executor.submit(_run_chunk, a) for a in batch]
                        for future in as_completed(futures):
                            all_results.append(future.result())
                            total_hands += _BATCH_HANDS_PER_WORKER
                            pbar.update(_BATCH_HANDS_PER_WORKER)

                num_states = sum(len(r["visit_count"]) for r in all_results)
                pbar.set_postfix(states=num_states)
        finally:
            pbar.close()
            num_states = save_merged(all_results, save_path)
            print(f"[完了] {total_hands} ハンド / {num_states} 状態 → {save_path}")
        if run_rare:
            # selfplay 時間の 20%（最低 1 分）
            rare_secs = max(60.0, max_seconds * 0.2)
            _run_rare_phase(num_players, save_path, num_simulations, num_workers, rare_secs)
        return

    # ── ハンド数ベースモード ─────────────────────────
    if effective_workers == 1:
        pbar = tqdm(total=num_hands, desc="Self-Play", unit="hands")
        chunk_size = max(num_hands // 20, 10)
        done = 0
        while done < num_hands:
            this_chunk = min(chunk_size, num_hands - done)
            result = _run_chunk((this_chunk, num_players, start_chips, num_simulations))
            all_results.append(result)
            done += this_chunk
            pbar.update(this_chunk)
            pbar.set_postfix(states=sum(len(r["visit_count"]) for r in all_results))
        pbar.close()

    else:
        base_chunk = num_hands // effective_workers
        chunks = [base_chunk] * effective_workers
        chunks[-1] += num_hands - sum(chunks)
        worker_args = [(c, num_players, start_chips, num_simulations) for c in chunks]

        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_run_chunk, a): i for i, a in enumerate(worker_args)}
            with tqdm(total=effective_workers, desc="Self-Play", unit="workers") as pbar:
                for future in as_completed(futures):
                    all_results.append(future.result())
                    pbar.update(1)
                    pbar.set_postfix(
                        done=f"{len(all_results) - 1}/{effective_workers}",
                        states=sum(len(r["visit_count"]) for r in all_results),
                    )

    num_states = save_merged(all_results, save_path)
    print(f"[完了] {num_states} 状態 → {save_path}")

    if run_rare:
        # ハンド数ベースの場合は固定 2 分
        _run_rare_phase(num_players, save_path, num_simulations, num_workers,
                        rare_secs=120.0)


def _run_rare_phase(
    num_players: int,
    save_path: str,
    num_simulations: int,
    num_workers: int,
    rare_secs: float,
) -> None:
    """selfplay 完了後に続けてレア状態の集中学習を実行する。"""
    minutes = rare_secs / 60
    print(f"\n── レア状態の集中学習を開始します（{minutes:.1f} 分間）──")
    run_rare_training(
        max_seconds=rare_secs,
        num_players=num_players,
        save_path=save_path,
        num_simulations=num_simulations,
        num_workers=num_workers,
        threshold=100,
    )


# ──────────────────────────────────────────────
# CLI エントリーポイント
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GTO CPU 事前学習スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 学習量の指定（どちらか一方）
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--hands", type=int, default=1000, metavar="N",
        help="学習ハンド数（--minutes と同時指定不可）",
    )
    time_group.add_argument(
        "--minutes", type=float, metavar="M",
        help="学習時間（分）。指定するとハンド数の代わりに時間で制御する",
    )

    parser.add_argument(
        "--players", type=int, default=4, choices=range(2, 11), metavar="N",
        help="テーブル人数 (2〜10)",
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
        "--workers", type=int, default=os.cpu_count() or 1, metavar="N",
        help="並列ワーカー数（デフォルト: 論理 CPU 数）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="ゲームの出力を表示する（シングルプロセスモードのみ有効）",
    )
    parser.add_argument(
        "--no-rare", action="store_true",
        help="selfplay 後のレア状態集中学習をスキップする",
    )
    args = parser.parse_args()

    run_selfplay(
        num_hands=None if args.minutes else args.hands,
        num_players=args.players,
        save_path=args.save,
        num_simulations=args.sims,
        num_workers=args.workers,
        verbose=args.verbose,
        max_seconds=args.minutes * 60 if args.minutes else None,
        run_rare=not args.no_rare,
    )


if __name__ == "__main__":
    main()


