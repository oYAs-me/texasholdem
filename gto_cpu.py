"""
gto_cpu.py

GTO (Game Theory Optimal) ベースの CPU エージェント。

既存の CpuAgent を継承し、decide_action のみを上書きする。
player.py / game.py への変更は不要。

使用方法:
    from gto_cpu import GtoCpu
    players.append(GtoCpu("GTO-CPU", chips=1000))
"""
from __future__ import annotations
import os
import random
from typing import Any

from player import CpuAgent
from card import Board
from fast_eval import calculate_equity_fast
from gto_strategy import (
    build_state_key, get_equity_bucket, get_street, heuristic_strategy,
    get_hand_potential, compute_action_values,
)
from gto_cfr import SimpleMCCFR
from hand_strength import evaluate_hand
import numpy as np

_DEFAULT_SAVE_PATH = "gto_strategy.json"

# ──────────────────────────────────────────────
# GTO 的なベットサイジング
# ──────────────────────────────────────────────
# ポットに対する倍率（CFR のアクション空間と 1:1 対応）
_RAISE_RATIOS: dict[str, float] = {
    'raise_33':  0.33,
    'raise_67':  0.67,
    'raise_100': 1.00,
    'raise_200': 2.00,
}
_RAISE_SIZES = list(_RAISE_RATIOS.keys())


def _expand_raise_actions(valid_actions: list[str]) -> list[str]:
    """game.py の 'raise' を CFR 用サイズ別アクションに展開する"""
    result = []
    for a in valid_actions:
        if a == 'raise':
            result.extend(_RAISE_SIZES)
        else:
            result.append(a)
    return result


def _compute_raise_amount(cfr_action: str, pot: int, min_raise: int) -> int:
    """サイズ別アクション名からベット額を計算する（10刻みに丸める）"""
    ratio = _RAISE_RATIOS[cfr_action]
    raw = int(pot * ratio)
    rounded = ((raw + 5) // 10) * 10
    return max(rounded, min_raise)


# ──────────────────────────────────────────────
# GTO CPU エージェント
# ──────────────────────────────────────────────

class GtoCpu(CpuAgent):
    """
    Regret Matching による混合戦略で動作する GTO CPU。

    初期状態: ヒューリスティック戦略（学習なしで即プレイ可能）
    ゲーム後: on_round_end() を呼ぶことで CFR 学習・JSON 保存ができる。
              ※ game.py を変更したくない場合は呼ばなくてもよい（ヒューリスティックで動作）
    """

    def __init__(
        self,
        name: str,
        chips: int,
        save_path: str = _DEFAULT_SAVE_PATH,
        num_simulations: int = 400,
        n_realtime: int = 20,
        load_path: str | None = None,
    ) -> None:
        super().__init__(name, chips)
        self._save_path = save_path
        self._num_simulations = num_simulations
        self._n_realtime = n_realtime
        self.cfr = SimpleMCCFR()
        self._rng = np.random.default_rng()

        effective_load = load_path if load_path is not None else save_path
        if effective_load != os.devnull:
            self.cfr.load(effective_load)

        # CFR 学習用の行動履歴（ラウンドごとにリセット）
        self._action_history: list[dict[str, Any]] = []

    def reset_for_new_round(self) -> None:
        super().reset_for_new_round()
        self._action_history.clear()

    # ──────────────────────────────────────────────
    # メイン意思決定
    # ──────────────────────────────────────────────

    def decide_action(
        self, valid_actions: list[str], game_state: dict[str, Any]
    ) -> tuple[str, int]:
        call_amount: int = game_state['call_amount']
        pot: int = game_state['pot']
        board: Board = game_state['board']
        min_raise: int = game_state.get('min_raise', call_amount * 2)
        max_raise_to: int = game_state.get('max_raise_to', self.chips + self.current_bet)

        num_opponents = len([
            p for p in game_state['players']
            if p['status'] in ('active', 'all-in') and p['name'] != self.name
        ])

        # ── エクイティ計算（高速評価器使用）──
        equity = 0.5
        if self.hand is not None:
            equity = calculate_equity_fast(
                self.hand, board, max(num_opponents, 1),
                num_simulations=self._num_simulations,
                rng=self._rng,
            )

        eq_bucket = get_equity_bucket(equity)
        is_last_to_act: bool = (game_state.get('last_to_act_name', '') == self.name)

        # ── ハンド・ポテンシャル詳細分類 ──
        hand_potential = 'na'
        street = get_street(board)
        if self.hand is not None:
            ev = evaluate_hand(self.hand, board)
            hand_potential = get_hand_potential(ev.hand_type, self.hand, board)

        state_key = build_state_key(
            equity, board, call_amount, pot, num_opponents,
            is_last_to_act=is_last_to_act, chips=self.chips,
            hand_potential=hand_potential,
            my_round_bet=getattr(self, 'round_bet', 0),
        )

        # ── CFR 用アクション空間: 'raise' をサイズ別に展開 ──
        cfr_actions = _expand_raise_actions(valid_actions)

        # ── リアルタイム再解決: 現在局面に特化した高速ロールアウトで regret を補強 ──
        if self._n_realtime > 0:
            self._realtime_resolve(
                state_key, cfr_actions, eq_bucket, equity, call_amount, pot,
                street=street, hand_potential=hand_potential,
            )

        # ── 混合戦略を取得（CFR or ヒューリスティック）──
        strategy = self.cfr.get_strategy(
            state_key, cfr_actions, eq_bucket,
            num_players=num_opponents + 1,
            call_amount=call_amount, pot=pot,
            street=street, hand_potential=hand_potential,
        )

        # チェック可能な状況でのフォールドは支配戦略（常にcheckの方が優位）
        if 'check' in cfr_actions and strategy.get('fold', 0.0) > 0.0:
            strategy['fold'] = 0.0
            total = sum(strategy.values())
            if total > 0:
                strategy = {a: v / total for a, v in strategy.items()}

        # ── fold が明らかに非合理なケースを強制排除 ──
        if call_amount > 0 and 'fold' in strategy:
            pot_odds = call_amount / (pot + call_amount)
            # オールインに近い（コール額 >= 自チップの 60%）かつ +EV
            is_near_allin = (call_amount >= self.chips * 0.6 and equity > 0.25)
            # ポットオッズよりエクイティが 10pt 以上高い（コールが明確に +EV）
            is_clear_call = (equity > pot_odds + 0.10)
            if is_near_allin or is_clear_call:
                strategy['fold'] = 0.0
                total = sum(strategy.values())
                if total > 0:
                    strategy = {a: v / total for a, v in strategy.items()}
                else:
                    # 全アクションが 0 になった場合: fold 以外を均等に配分
                    non_fold = [a for a in strategy if a != 'fold']
                    if non_fold:
                        w = 1.0 / len(non_fold)
                        strategy = {a: (w if a != 'fold' else 0.0) for a in strategy}
                    # else: fold しか選択肢がない（そのままにする）

        # ── 確率的にアクションを選択（GTOの核心）──
        # ウェイト合計が 0 の場合（理論上は起きないが念のため）は均等分配にフォールバック
        weights = list(strategy.values())
        if sum(weights) <= 0:
            weights = [1.0] * len(weights)
        cfr_action = random.choices(list(strategy.keys()), weights=weights)[0]

        # ── game.py 用に逆変換・ベットサイズ計算 ──
        amount = 0
        if cfr_action == 'call':
            amount = call_amount
            game_action = 'call'
        elif cfr_action in _RAISE_RATIOS:
            amount = _compute_raise_amount(cfr_action, pot, min_raise)
            # 相手が出せる最大額を超えないよう上限を設定（無駄なオーバーレイズ防止）
            amount = min(amount, max_raise_to)
            game_action = 'raise'
        else:
            game_action = cfr_action  # 'fold' or 'check'

        # 行動履歴を保存（CFR 更新用: cfr_action でサイズ別後悔を蓄積）
        self._action_history.append({
            'state_key': state_key,
            'eq_bucket': eq_bucket,
            'strategy': strategy,
            'action': cfr_action,
            'call_amount': call_amount,
            'pot': pot,
            'equity': equity,
            'street': street,
            'hand_potential': hand_potential,
        })

        return game_action, amount

    # ──────────────────────────────────────────────
    # リアルタイム再解決
    # ──────────────────────────────────────────────

    def _realtime_resolve(
        self,
        state_key: str,
        cfr_actions: list[str],
        eq_bucket: int,
        equity: float,
        call_amount: int,
        pot: int,
        street: str = 'postflop',
        hand_potential: str = 'na',
    ) -> None:
        """
        現在局面に特化した高速ロールアウトで regret_sum / strategy_sum を補強する。
        """
        for _ in range(self._n_realtime):
            # 現在の後悔から現時点の戦略を計算（visit_count を触らない）
            regrets = self.cfr.regret_sum.get(state_key, {})
            positive = {a: max(regrets.get(a, 0.0), 0.0) for a in cfr_actions}
            total_r = sum(positive.values())
            if total_r > 0:
                strategy = {a: positive[a] / total_r for a in cfr_actions}
            else:
                strategy = heuristic_strategy(
                    eq_bucket, cfr_actions,
                    call_amount=call_amount, pot=pot,
                    street=street, hand_potential=hand_potential,
                )

            # エクイティに基づく確率的ロールアウト
            won = random.random() < equity
            reward = 1.0 if won else -1.0

            # 戦略に従ってアクションをサンプリング
            taken = random.choices(
                list(strategy.keys()), weights=list(strategy.values())
            )[0]

            action_values = compute_action_values(
                reward, equity, taken, cfr_actions,
                call_amount, pot, street=street, hand_potential=hand_potential,
            )

            self.cfr.update_regret(state_key, taken, action_values)
            # realtime resolve の学習も平均戦略に反映する
            self.cfr.update_strategy_sum(state_key, strategy)

    # ──────────────────────────────────────────────
    # CFR 学習フック（オプション）
    # ──────────────────────────────────────────────

    def on_round_end(self, won: bool) -> None:
        """
        ラウンド終了後に呼ぶと CFR 学習を行い戦略 JSON を更新する。
        game.py を変更したくない場合は呼ばなくてよい（ヒューリスティックで動作）。

        game.py の showdown / end_round_early の直後に追加可能:
            for p in game.players:
                if isinstance(p, GtoCpu):
                    p.on_round_end(p in winners)
        """
        reward = 1.0 if won else -1.0

        for record in self._action_history:
            state_key = record['state_key']
            strategy = record['strategy']
            taken = record['action']
            equity_rec: float = record.get('equity', 0.5)
            call_amount_rec: int = record.get('call_amount', 0)
            pot_rec: int = record.get('pot', 1)
            street_rec: str = record.get('street', 'postflop')
            hand_pot_rec: str = record.get('hand_potential', 'na')

            action_values = compute_action_values(
                reward, equity_rec, taken, list(strategy.keys()),
                call_amount_rec, pot_rec,
                street=street_rec, hand_potential=hand_pot_rec,
            )

            self.cfr.update_regret(state_key, taken, action_values)

        self._action_history.clear()
        self.cfr.save(self._save_path)
