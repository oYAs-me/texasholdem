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

from card import Hand, Board, create_deck
from gto_cpu import GtoCpu
from gto_cfr import SimpleMCCFR
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

    start_round() で target_textures に合致するデッキを
    リジェクションサンプリングで生成する。
    game.py の start_round をベースにデッキ生成部分だけ変更。
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

    def _make_biased_deck(self, num_contesting: int) -> list:
        """
        target_textures に合致するフロップが来るデッキを生成。

        デッキは末尾からポップされるため、フロップカードの位置は:
            deck[-(num_contesting*2 + 3) : -(num_contesting*2)]
        最大 100 回試行し、合致しなければランダムデッキにフォールバック。
        """
        if not self._target_textures:
            deck = create_deck()
            random.shuffle(deck)
            return deck

        n = num_contesting * 2  # ホールカード分のオフセット
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
        deck = create_deck()
        random.shuffle(deck)
        return deck

    def start_round(self) -> bool:
        """
        game.py の start_round をほぼ踏襲し、デッキ生成部分のみ偏り付きに変更。
        出力は selfplay 時に redirect_stdout で抑制されるため print 省略。
        """
        self.board = Board()
        self.pot = 0
        self.current_bet = 0

        for p in self.players:
            p.reset_for_new_round()

        contesting = self.get_contesting_players()
        if len(contesting) < 2:
            return False

        # ブラインド位置の決定
        sb_pos = (self.dealer_pos + 1) % len(self.players)
        while self.players[sb_pos].status == 'busted':
            sb_pos = (sb_pos + 1) % len(self.players)
        bb_pos = (sb_pos + 1) % len(self.players)
        while self.players[bb_pos].status == 'busted':
            bb_pos = (bb_pos + 1) % len(self.players)

        # ブラインド支払い
        sb_amount = self.players[sb_pos].pay(self.small_blind)
        bb_amount = self.players[bb_pos].pay(self.big_blind)
        self.pot += sb_amount + bb_amount
        self.current_bet = self.big_blind

        # ── 偏ったデッキを生成してホールカードをディール ──
        self.deck = self._make_biased_deck(len(contesting))
        for p in contesting:
            p.hand = Hand((self.deck.pop(), self.deck.pop()))

        return True


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
    merged_regret:   dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    merged_strategy: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    merged_visits:   dict[str, int] = defaultdict(int)

    for p in players:
        for k, v in p.cfr.regret_sum.items():
            if k not in known_common_states:
                for a, val in v.items():
                    merged_regret[k][a] += val
        for k, v in p.cfr.strategy_sum.items():
            if k not in known_common_states:
                for a, val in v.items():
                    merged_strategy[k][a] += val
        for k, val in p.cfr.visit_count.items():
            if k not in known_common_states:
                merged_visits[k] += val

    return {
        "regret_sum":   {k: dict(v) for k, v in merged_regret.items()},
        "strategy_sum": {k: dict(v) for k, v in merged_strategy.items()},
        "visit_count":  dict(merged_visits),
    }


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
            parts = state_key.split(':')
            # preflop はボードがないためテクスチャは 'na'; スキップ
            if len(parts) >= 3 and parts[0] != 'preflop':
                texture_counter[parts[2]] += 1

    target_textures = frozenset(texture_counter.keys())
    summary = {
        "rare": rare_count,
        "common": len(known_common),
        "textures": dict(sorted(texture_counter.items(), key=lambda x: -x[1])),
    }
    return frozenset(known_common), target_textures, summary


# ──────────────────────────────────────────────
# CFR データのマージ / 保存（gto_selfplay.py と同じパターン）
# ──────────────────────────────────────────────

def _merge_cfr_data(results: list[dict]) -> dict:
    regret:   dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    strategy: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    visits:   dict[str, int] = defaultdict(int)

    for result in results:
        for state, actions in result["regret_sum"].items():
            for action, value in actions.items():
                regret[state][action] += value
        for state, actions in result["strategy_sum"].items():
            for action, value in actions.items():
                strategy[state][action] += value
        for state, count in result["visit_count"].items():
            visits[state] += count

    return {
        "regret_sum":   {k: dict(v) for k, v in regret.items()},
        "strategy_sum": {k: dict(v) for k, v in strategy.items()},
        "visit_count":  dict(visits),
    }


def _load_base(save_path: str) -> dict:
    base_cfr = SimpleMCCFR()
    base_cfr.load(save_path)
    return {
        "regret_sum":   {k: dict(v) for k, v in base_cfr.regret_sum.items()},
        "strategy_sum": {k: dict(v) for k, v in base_cfr.strategy_sum.items()},
        "visit_count":  dict(base_cfr.visit_count),
    }


def _save_merged(results: list[dict], save_path: str) -> int:
    merged = _merge_cfr_data(results)
    final_cfr = SimpleMCCFR()
    for k, v in merged["regret_sum"].items():
        final_cfr.regret_sum[k] = defaultdict(float, v)
    for k, v in merged["strategy_sum"].items():
        final_cfr.strategy_sum[k] = defaultdict(float, v)
    for k, v in merged["visit_count"].items():
        final_cfr.visit_count[k] = v
    final_cfr.save(save_path)
    return len(merged["visit_count"])


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

    base_data = _load_base(save_path)
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
        num_states = _save_merged(all_results, save_path)
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
        "--players", type=int, default=4, choices=range(2, 7), metavar="N",
        help="テーブル人数 (2〜6)",
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
