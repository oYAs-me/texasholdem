"""
gto_rare_training.py

珍しい状況（visit_count が低い state_key）に特化した学習スクリプト。

2 段階のアプローチ:
  1. ボードテクスチャのリジェクションサンプリング:
     レア状態のテクスチャに合致するボードを優先的に生成し、
     ランダム selfplay より高頻度でレア状態に到達させる。
  2. CFR 更新フィルタリング:
     visit_count < threshold の状態（+ 未学習状態）のみ CFR に反映。
     通常 selfplay で学習済みの状態は汚染しない。

使用例:
    uv run python gto_rare_training.py                     # 2000 ハンド
    uv run python gto_rare_training.py --minutes 30        # 30 分間
    uv run python gto_rare_training.py --threshold 200 --workers 4
    uv run python gto_rare_training.py --hands 5000 --sims 100
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from card import Board, create_deck
from gto_cpu import GtoCpu
from gto_cfr import SimpleMCCFR
from gto_cfr_utils import merge_cfr_data, load_base, save_merged, cfr_to_dict
from gto_strategy import classify_board_texture
from learning_game import LearningGame

_RARE_THRESHOLD_DEFAULT = 100
_BATCH_HANDS_PER_WORKER = 50


# ──────────────────────────────────────────────
# 偏りつきゲームクラス
# ──────────────────────────────────────────────

class BiasedLearningGame(LearningGame):
    """
    ボードテクスチャを偏らせた LearningGame。

    game.py の _make_deck() をオーバーライドして、target_textures に合致する
    デッキをリジェクションサンプリングで生成する。
    """

    def __init__(
        self,
        players,
        start_chips: int = 1000,
        sb: int = 10,
        bb: int = 20,
        target_textures: frozenset[str] | None = None,
    ) -> None:
        super().__init__(players, start_chips=start_chips, sb=sb, bb=bb)
        self._target_textures: frozenset[str] = target_textures or frozenset()

    def _make_deck(self, contesting: list) -> list:
        """
        target_textures に合致するフロップが来るデッキを生成。

        デッキは末尾からポップされるため、フロップカードの位置は:
            deck[-(num_contesting*2 + 3) : -(num_contesting*2)]
        最大 100 回試行し、合致しなければランダムデッキにフォールバック。
        """
        if not self._target_textures:
            return super()._make_deck(contesting)

        n = len(contesting) * 2  # ホールカード分のオフセット
        for _ in range(100):
            deck = create_deck()
            random.shuffle(deck)
            # フロップカード候補（末尾から n+3 番目〜 n+1 番目）
            flop_slice = deck[-(n + 3) : (-n if n > 0 else len(deck))]
            if len(flop_slice) == 3:
                tmp_board = Board()
                tmp_board.set_flops(tuple(flop_slice))
                texture = classify_board_texture(tmp_board)
                if texture in self._target_textures:
                    return deck

        # フォールバック: ランダム
        return super()._make_deck(contesting)


# ──────────────────────────────────────────────
# ワーカー関数（トップレベル: Windows spawn で pickle 可能にするため）
# ──────────────────────────────────────────────

def _run_rare_chunk(args: tuple) -> dict:
    """
    サブプロセスで実行される学習チャンク。
    known_common_states に含まれない state_key（レア / 未学習）の
    CFR データのみを返す。
    """
    chunk_hands, num_players, start_chips, num_simulations, known_common_states, target_textures = args

    players = [
        GtoCpu(
            f"GTO-{i}",
            chips=start_chips,
            save_path=os.devnull,
            num_simulations=num_simulations,
            n_realtime=0,  # selfplay では無効化（速度優先）
        )
        for i in range(num_players)
    ]
    game = BiasedLearningGame(
        players,
        start_chips=start_chips,
        sb=10,
        bb=20,
        target_textures=target_textures,
    )

    for _ in range(chunk_hands):
        active = [p for p in game.players if p.chips > 0]
        if len(active) < 2:
            for p in game.players:
                p.chips = start_chips
                p.status = "active"
        with contextlib.redirect_stdout(io.StringIO()):
            game.play_round()

    # レア状態 / 未学習状態のみをマージ
    filtered_results = []
    for p in players:
        d = cfr_to_dict(p.cfr)
        filtered = {
            "regret_sum":   {k: v for k, v in d["regret_sum"].items()   if k not in known_common_states},
            "strategy_sum": {k: v for k, v in d["strategy_sum"].items() if k not in known_common_states},
            "visit_count":  {k: v for k, v in d["visit_count"].items()  if k not in known_common_states},
        }
        filtered_results.append(filtered)
    return merge_cfr_data(filtered_results)


# ──────────────────────────────────────────────
# レア状態の分析
# ──────────────────────────────────────────────

def _analyze_rare_states(
    save_path: str, threshold: int
) -> tuple[frozenset[str], frozenset[str], dict]:
    """
    gto_strategy.json を読み込み、レア状態を分析する。

    Returns:
        known_common_states: visit_count >= threshold の状態（CFR 更新スキップ対象）
        target_textures:     レア状態が含むボードテクスチャ（biased deck 用）
        summary:             ログ出力用サマリ dict
    """
    base_cfr = SimpleMCCFR()
    base_cfr.load(save_path)

    known_common: set[str] = set()
    texture_counter: dict[str, int] = defaultdict(int)
    rare_count = 0

    for state_key, count in base_cfr.visit_count.items():
        if count >= threshold:
            known_common.add(state_key)
        else:
            rare_count += 1
            parts = state_key.split('_')
            # preflop はボードがないためテクスチャは 'na'; スキップ
            if len(parts) >= 2 and parts[0] != 'preflop':
                texture_counter[parts[1]] += 1

    target_textures = frozenset(texture_counter.keys())
    summary = {
        "rare": rare_count,
        "common": len(known_common),
        "textures": dict(sorted(texture_counter.items(), key=lambda x: -x[1])),
    }
    return frozenset(known_common), target_textures, summary

# ──────────────────────────────────────────────
# メイン学習ループ
# ──────────────────────────────────────────────

def run_rare_training(
    max_seconds: float,
    num_players: int,
    save_path: str,
    num_simulations: int,
    num_workers: int,
    threshold: int,
) -> None:
    # ── レア状態の分析 ──
    known_common, target_textures, summary = _analyze_rare_states(save_path, threshold)
    print(f"レア状態分析 (threshold={threshold})")
    print(f"  学習済み（スキップ）: {summary['common']} 状態")
    print(f"  レア（ターゲット）  : {summary['rare']} 状態")
    print(f"  ターゲットテクスチャ: {summary['textures']}")

    if not target_textures:
        print("  ※ ターゲットテクスチャなし（全状態がプリフロップ or 空）: ランダムデッキで実行")

    start_chips = 1000
    effective_workers = num_workers
    minutes = max_seconds / 60

    print(
        f"\n珍しい状況の集中学習: {num_players} 人 × {int(minutes)}分間"
        f" | workers={effective_workers}"
        f" | sims={num_simulations}"
        f" → {save_path}"
    )

    base_data = load_base(save_path)
    all_results: list[dict] = [base_data]
    chunk_args = (
        _BATCH_HANDS_PER_WORKER,
        num_players,
        start_chips,
        num_simulations,
        known_common,
        target_textures,
    )

    end_time = time.monotonic() + max_seconds
    total_hands = 0
    pbar = tqdm(
        desc="RareTraining",
        unit="hands",
        bar_format="{desc}: {n_fmt} hands [{elapsed}, {rate_fmt}]{postfix}",
    )
    try:
        while time.monotonic() < end_time:
            if effective_workers == 1:
                result = _run_rare_chunk(chunk_args)
                all_results.append(result)
                total_hands += _BATCH_HANDS_PER_WORKER
                pbar.update(_BATCH_HANDS_PER_WORKER)
            else:
                batch = [chunk_args] * effective_workers
                with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                    futures = [executor.submit(_run_rare_chunk, a) for a in batch]
                    for future in as_completed(futures):
                        all_results.append(future.result())
                        total_hands += _BATCH_HANDS_PER_WORKER
                        pbar.update(_BATCH_HANDS_PER_WORKER)
            rare_updated = sum(len(r["visit_count"]) for r in all_results[1:])
            pbar.set_postfix(rare_updated=rare_updated)
    finally:
        pbar.close()
        num_states = save_merged(all_results, save_path)
        print(f"[完了] 学習完了: {total_hands} ハンド / {num_states} 状態 → {save_path}")


# ──────────────────────────────────────────────
# CLI エントリーポイント
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="レア状態に特化した GTO 集中学習スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--minutes", type=float, default=5.0, metavar="M",
        help="学習時間（分）",
    )
    parser.add_argument(
        "--threshold", type=int, default=_RARE_THRESHOLD_DEFAULT, metavar="N",
        help="visit_count がこの値未満の状態をレアとみなす",
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
        help="エクイティ計算の Monte Carlo 試行数",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1, metavar="N",
        help="並列ワーカー数（デフォルト: 論理 CPU 数）",
    )
    args = parser.parse_args()

    run_rare_training(
        max_seconds=args.minutes * 60,
        num_players=args.players,
        save_path=args.save,
        num_simulations=args.sims,
        num_workers=args.workers,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
