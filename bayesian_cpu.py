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
from fast_eval import calculate_equity_fast
from bayesian_strategy import BeliefTracker, PlayerProfile, hand_to_group


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
        min_raise: int = game_state.get('min_raise', max(call_amount * 2, 1))

        active_opponents = [
            p for p in game_state['players']
            if p['status'] in ('active', 'all-in') and p['name'] != self.name
        ]
        num_opponents = len(active_opponents)

        # ── エクイティ計算（高速モンテカルロ）──
        equity = 0.5
        if self.hand is not None and num_opponents > 0:
            equity = calculate_equity_fast(
                self.hand, board, num_opponents,
                num_simulations=self._num_simulations,
                rng=self._rng,
            )

        # ── レンジ推定によるエクイティ補正 ──
        if self._trackers and active_opponents:
            adj = self._compute_range_adjustment(active_opponents)
            equity = float(np.clip(equity - adj, 0.05, 0.95))

        pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 else 0.0
        return self._bayesian_action(equity, pot_odds, valid_actions, game_state)

    # ── 内部ヘルパー ─────────────────────────────────────────────────────

    def _compute_range_adjustment(self, active_opponents: list[dict]) -> float:
        """
        対戦相手レンジの平均強度からエクイティへの調整値を計算する。

        プロファイルが利用可能な場合、ショーダウン実績を追加情報として加味する。

        返り値: float
            正 → 相手レンジが強い → 自エクイティを下げる
            負 → 相手レンジが弱い → 自エクイティを上げる
        """
        total = 0.0
        counted = 0
        for opp in active_opponents:
            name = opp['name']
            # ベリーフトラッカーが存在する場合（当ラウンドでアクションを観測済み）
            if name in self._trackers:
                mean_s = self._trackers[name].mean_strength()
            elif name in self._profiles and self._profiles[name].has_profile:
                # まだアクションを見ていないが過去の統計がある場合:
                # ショーダウン実績の平均強度を基に調整
                mean_s = self._profiles[name].avg_showdown_strength
            else:
                continue
            total += (mean_s - 0.5) * 0.25
            counted += 1
        return total / counted if counted > 0 else 0.0

    def _bayesian_action(
        self,
        equity: float,
        pot_odds: float,
        valid_actions: list[str],
        game_state: dict,
    ) -> tuple[str, int]:
        """調整済みエクイティに基づくアクション決定（バランスド寄りのスタイル）。"""
        call_amount = game_state['call_amount']
        pot = game_state['pot']
        min_raise = game_state.get('min_raise', max(call_amount * 2, 1))

        if equity > 0.72 and 'raise' in valid_actions:
            size = self._get_dynamic_raise_size(pot, min_raise, equity, 1.1)
            return 'raise', size

        if equity > pot_odds + 0.07 or call_amount == 0:
            if equity > 0.60 and 'raise' in valid_actions:
                size = self._get_dynamic_raise_size(pot, min_raise, equity, 0.9)
                return 'raise', size
            return ('call' if 'call' in valid_actions else 'check'), call_amount

        return 'fold', 0

