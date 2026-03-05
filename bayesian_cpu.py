"""
bayesian_cpu.py

ベイジアン手札範囲推定を利用する CPU エージェント。

各対戦相手の BeliefTracker（ラウンド内）と PlayerProfile（ラウンドを跨いで永続）を管理し、
観測したアクション（fold 含む）でベイズ更新を行う。

fold が多いプレイヤーへの対応:
    - PlayerProfile が VPIP / AF を複数ラウンドで蓄積する
    - タイトなプレイヤー（低VPIP）が fold したときの尤度を強くする
      → 「彼は弱い手しか fold しない」 という知識をモデルに反映
    - ショーダウン時に実際の手牌を観測して ground truth として記録する
    - fold したラウンドもベイズ更新を行い、その情報を当ラウンドの信念に活かす

game.py の betting_round での on_opponent_action() と
showdown() での on_showdown_hand() が呼ばれることを前提とする。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from player import CpuAgent
from card import Board, Hand
from fast_eval import calculate_equity_fast, card_to_int
from bayesian_strategy import (
    BeliefTracker, PlayerProfile, hand_to_group, calculate_equity_vs_range,
)


class BayesianCpu(CpuAgent):
    """
    ベイジアン手札範囲推定による意思決定 CPU。

    _profiles: dict[str, PlayerProfile]
        ラウンドを跨いで永続するプレイヤー行動統計。
        VPIP / PFR / AF / ショーダウン実績を蓄積し、
        尤度関数の個人化（fold 含む各アクションの解釈精度向上）に使う。

    _trackers: dict[str, BeliefTracker]
        ラウンド内の手牌分布ベリーフ。ラウンド終了時にリセット。

    _preflop_seen: set[str]
        当ラウンドで最初のプリフロップアクションを記録済みのプレイヤー名。
        プロファイルの hands_dealt を正確に1回だけ加算するため。
    """

    def __init__(self, name: str, chips: int, num_simulations: int = 300) -> None:
        super().__init__(name, chips)
        self._num_simulations = num_simulations
        self._profiles: dict[str, PlayerProfile] = {}   # 永続
        self._trackers: dict[str, BeliefTracker] = {}   # ラウンドごとにリセット
        self._preflop_seen: set[str] = set()             # ラウンドごとにリセット

    def reset_for_new_round(self) -> None:
        super().reset_for_new_round()
        self._trackers.clear()
        self._preflop_seen.clear()
        # _profiles は保持（ラウンドを跨いで学習し続ける）

    # ── 観測フック ───────────────────────────────────────────────────────

    def on_opponent_action(
        self,
        player_name: str,
        action: str,
        amount: int,
        game_state: dict[str, Any],
    ) -> None:
        """
        対戦相手のアクションを観測してプロファイルとベリーフを更新する。

        fold したプレイヤーも更新対象とする。
        「fold した」という事実は、手牌についての重要な情報を持つ。

        amount: 実際に賭けた額
            fold / check → 0
            call         → call_amount
            raise        → raise_to_amount (decide_action の戻り値)
        """
        board: Board = game_state['board']
        call_amount: int = game_state['call_amount']
        pot: int = game_state['pot']

        # プロファイルの初期化
        if player_name not in self._profiles:
            self._profiles[player_name] = PlayerProfile()
        profile = self._profiles[player_name]

        board_cards = board.get_all_cards()
        is_preflop = len(board_cards) == 0

        # ── プロファイル更新 ────────────────────────────────────────────
        if is_preflop and player_name not in self._preflop_seen:
            profile.record_preflop_action(action)
            self._preflop_seen.add(player_name)
        elif not is_preflop:
            profile.record_postflop_action(action)

        # ── ベリーフ更新（fold も含めて行う）──────────────────────────
        if player_name not in self._trackers:
            self._trackers[player_name] = BeliefTracker()
        tracker = self._trackers[player_name]

        # raise の場合は amount（raise-to 額）、それ以外は call_amount を使う
        bet_amount = amount if action == 'raise' else call_amount

        if is_preflop:
            tracker.update(action, bet_amount, pot, profile=profile)
        else:
            dead = list(board_cards)
            if self.hand:
                dead.extend(self.hand.cards)
            if not tracker.is_postflop:
                tracker.expand_to_postflop(dead)
            tracker.update(action, bet_amount, pot, dead_cards=dead, profile=profile)

    def on_showdown_hand(self, player_name: str, hand: Hand) -> None:
        """
        ショーダウン時に相手の実際の手牌を観測してプロファイルに記録する。

        これにより:
        - 長期的に「このプレイヤーはどの強さの手でショーダウンするか」が蓄積される
        - fold が多いプレイヤーのショーダウン実績は VPIP 推定の補正に使える
        """
        if player_name not in self._profiles:
            self._profiles[player_name] = PlayerProfile()
        c1, c2 = hand.cards
        group_idx = hand_to_group(c1, c2)
        self._profiles[player_name].record_showdown_hand(group_idx)

    # ── 意思決定 ──────────────────────────────────────────────────────────

    def decide_action(
        self, valid_actions: list[str], game_state: dict[str, Any]
    ) -> tuple[str, int]:
        call_amount: int = game_state['call_amount']
        pot: int = game_state['pot']
        board: Board = game_state['board']

        active_opponents = [
            p for p in game_state['players']
            if p['status'] in ('active', 'all-in') and p['name'] != self.name
        ]
        num_opponents = len(active_opponents)

        # ── エクイティ計算 ──────────────────────────────────────────────
        # トラッカーがある相手には推定レンジに対して直接エクイティを計算する
        equity = 0.5
        if self.hand is not None and num_opponents > 0:
            dead_ints: set[int] = {card_to_int(c) for c in self.hand.cards}
            dead_ints.update(card_to_int(c) for c in board.get_all_cards())

            tracker_weights = []
            has_tracker = False
            for opp in active_opponents:
                name = opp['name']
                if name in self._trackers:
                    tracker_weights.append(
                        self._trackers[name].get_combo_weights(dead_ints)
                    )
                    has_tracker = True
                else:
                    tracker_weights.append(None)

            if has_tracker:
                equity = calculate_equity_vs_range(
                    self.hand, board, tracker_weights,
                    num_simulations=self._num_simulations, rng=self._rng,
                )
            else:
                equity = calculate_equity_fast(
                    self.hand, board, num_opponents,
                    num_simulations=self._num_simulations, rng=self._rng,
                )

        pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 else 0.0
        return self._bayesian_action(
            equity, pot_odds, valid_actions, game_state, active_opponents
        )

    # ── 内部ヘルパー ─────────────────────────────────────────────────────

    def _mean_opponent_strength(self, active_opponents: list[dict]) -> float:
        """
        アクティブな対戦相手のレンジ平均強度を返す (0=弱〜1=強)。

        ベリーフトラッカーが存在する場合はその値を使い、
        なければ過去のショーダウン実績から推定する。
        """
        total = 0.0
        counted = 0
        for opp in active_opponents:
            name = opp['name']
            if name in self._trackers:
                total += self._trackers[name].mean_strength()
                counted += 1
            elif name in self._profiles and self._profiles[name].has_profile:
                total += self._profiles[name].avg_showdown_strength
                counted += 1
        return total / counted if counted > 0 else 0.5

    def _estimate_fold_equity(
        self, mean_strength: float, raise_size: int, pot: int, call_amount: int = 0
    ) -> float:
        """
        相手レンジの強度とレイズサイズから、相手がフォールドする確率を推定する。

        - 弱いレンジほどフォールドしやすい (1 - mean_strength)
        - ポットに対して大きいベットほどフォールドを誘発しやすい
        - call_amount > 0 の場合（相手がすでにレイズ済み）はpot commitmentで
          フォールドしにくくなる。call_amount/(pot+call_amount) の割合で割引。
        上限を 0.75 に設定し（ナッツは絶対フォールドしない）過大推定を防ぐ。
        """
        pot_fraction = raise_size / max(pot, 1)
        raw = (1.0 - mean_strength) * (0.30 + 0.40 * min(pot_fraction, 1.5))
        # Pot commitment discount: 相手がすでにレイズしているほどfold_eqを減らす
        if call_amount > 0:
            commitment = call_amount / max(pot + call_amount, 1)
            raw *= (1.0 - commitment)
        return float(np.clip(raw, 0.0, 0.75))

    def _bayesian_action(
        self,
        equity: float,
        pot_odds: float,
        valid_actions: list[str],
        game_state: dict,
        active_opponents: list[dict] | None = None,
    ) -> tuple[str, int]:
        """
        各アクションの期待値（EV）を算出し、最大EVのアクションを選択する。

        EV計算モデル（多人数対応）:
            EV(Fold)  = 0
            EV(Call)  = equity × (pot + call) - call
            EV(Check) = equity × pot
            EV(Raise to R) = fold_eq_all × pot
                             + (1 - fold_eq_all) × [equity × total_pot - R]
            fold_eq_all = fold_eq_per ^ N  （N = 相手人数）
        """
        call_amount = game_state['call_amount']
        pot = game_state['pot']
        min_raise = game_state.get('min_raise', max(call_amount * 2, 1))
        effective_pot = pot + call_amount

        num_opps = len(active_opponents) if active_opponents else 1
        mean_s = self._mean_opponent_strength(active_opponents or [])

        # ── EV(Fold) ──────────────────────────────────────────────────
        ev_fold = 0.0

        # ── EV(Call / Check) ──────────────────────────────────────────
        if call_amount > 0:
            ev_call = equity * (pot + call_amount) - call_amount
            action_call = 'call'
        else:
            ev_call = equity * pot
            action_call = 'check' if 'check' in valid_actions else 'call'

        # ── EV(Bet / Raise) — 複数サイズを評価して最善を選ぶ ──────────
        best_ev_raise = float('-inf')
        best_raise_size = min_raise

        if 'raise' in valid_actions:
            for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
                size = max(min_raise, int(effective_pot * factor))
                size = min(size, self.chips)
                size = max(((size + 5) // 10) * 10, min_raise)

                # Kelly基準: f* = equity - (1-equity) × bet/pot
                # 多人数ほど不確実性が高く 1/N で割引（分散管理）
                kelly_f = equity - (1.0 - equity) * size / max(pot, 1)
                kelly_cap = int(kelly_f * self.chips / num_opps) if kelly_f > 0 else 0
                if kelly_cap < min_raise:
                    continue  # Kellyフラクションが小さすぎてmin_raiseに届かない
                size = min(size, kelly_cap)
                size = max(((size + 5) // 10) * 10, min_raise)

                # fold_eq: call_amount を渡してpot commitment discountを適用
                fold_eq_per = self._estimate_fold_equity(mean_s, size, pot, call_amount)
                fold_eq_all = fold_eq_per ** num_opps
                exp_callers = num_opps * (1.0 - fold_eq_per)
                total_pot_called = pot + size + exp_callers * max(0.0, size - call_amount)
                ev = fold_eq_all * pot + (1.0 - fold_eq_all) * (equity * total_pot_called - size)

                if ev > best_ev_raise:
                    best_ev_raise = ev
                    best_raise_size = size

        # ── 最大 EV のアクションを選択 ────────────────────────────────
        candidates: list[tuple[str, int, float]] = [('fold', 0, ev_fold)]
        if action_call in valid_actions:
            candidates.append((action_call, call_amount, ev_call))
        if 'raise' in valid_actions:
            candidates.append(('raise', best_raise_size, best_ev_raise))

        best_action, best_amount, _ = max(candidates, key=lambda x: x[2])
        return best_action, best_amount

