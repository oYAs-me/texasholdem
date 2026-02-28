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
import random
from typing import Any

from player import CpuAgent
from card import Board
from probability import calculate_equity
from gto_strategy import build_state_key, get_equity_bucket, get_street, heuristic_strategy
from gto_cfr import SimpleMCCFR
from hand_strength import evaluate_hand

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
    ) -> None:
        super().__init__(name, chips)
        self._save_path = save_path
        self._num_simulations = num_simulations
        self._n_realtime = n_realtime
        self.cfr = SimpleMCCFR()
        self.cfr.load(save_path)

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

        num_opponents = len([
            p for p in game_state['players']
            if p['status'] in ('active', 'all-in') and p['name'] != self.name
        ])

        # ── エクイティ計算（既存モジュールを流用）──
        equity = 0.5
        if self.hand is not None:
            equity = calculate_equity(
                self.hand, board, max(num_opponents, 1),
                num_simulations=self._num_simulations,
            )

        eq_bucket = get_equity_bucket(equity)
        is_last_to_act: bool = (game_state.get('last_to_act_name', '') == self.name)

        # ── ハンド・ポテンシャル詳細分類 ──
        hand_potential = 'na'
        if self.hand is not None:
            from gto_strategy import get_hand_potential
            ev = evaluate_hand(self.hand, board)
            hand_potential = get_hand_potential(ev.hand_type, self.hand, board)

        state_key = build_state_key(
            equity, board, call_amount, pot, num_opponents,
            is_last_to_act=is_last_to_act, chips=self.chips,
            hand_potential=hand_potential,
        )

        # ── CFR 用アクション空間: 'raise' をサイズ別に展開 ──
        cfr_actions = _expand_raise_actions(valid_actions)

        # ── リアルタイム再解決: 現在局面に特化した高速ロールアウトで regret を補強 ──
        if self._n_realtime > 0:
            self._realtime_resolve(state_key, cfr_actions, eq_bucket, equity, call_amount, pot)

        # ── 混合戦略を取得（CFR or ヒューリスティック）──
        strategy = self.cfr.get_strategy(state_key, cfr_actions, eq_bucket)

        # チェック可能な状況でのフォールドは支配戦略（常にcheckの方が優位）
        if 'check' in cfr_actions and strategy.get('fold', 0.0) > 0.0:
            strategy['fold'] = 0.0
            total = sum(strategy.values())
            if total > 0:
                strategy = {a: v / total for a, v in strategy.items()}

        # ── 確率的にアクションを選択（GTOの核心）──
        cfr_action = random.choices(
            list(strategy.keys()), weights=list(strategy.values())
        )[0]

        # ── game.py 用に逆変換・ベットサイズ計算 ──
        amount = 0
        if cfr_action == 'call':
            amount = call_amount
            game_action = 'call'
        elif cfr_action in _RAISE_RATIOS:
            amount = _compute_raise_amount(cfr_action, pot, min_raise)
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
    ) -> None:
        """
        現在局面に特化した高速ロールアウトで regret_sum を補強する。

        すでに計算済みの equity を確率的勝敗のシードとして使い、
        追加 Monte Carlo コストなしで n_realtime 回の反実仮想更新を実行。
        visit_count は変更しないため selfplay ログには影響しない。
        """
        can_check = 'check' in cfr_actions
        total_pot = pot + call_amount
        fold_fraction = call_amount / total_pot if total_pot > 0 else 0.3

        for _ in range(self._n_realtime):
            # 現在の後悔から現時点の戦略を計算（visit_count を触らない）
            regrets = self.cfr.regret_sum.get(state_key, {})
            positive = {a: max(regrets.get(a, 0.0), 0.0) for a in cfr_actions}
            total_r = sum(positive.values())
            if total_r > 0:
                strategy = {a: positive[a] / total_r for a in cfr_actions}
            else:
                strategy = heuristic_strategy(eq_bucket, cfr_actions)

            # エクイティに基づく確率的ロールアウト
            won = random.random() < equity
            reward = 1.0 if won else -1.0

            # 戦略に従ってアクションをサンプリング
            taken = random.choices(
                list(strategy.keys()), weights=list(strategy.values())
            )[0]

            # 各アクションの反実仮想価値を推定
            action_values: dict[str, float] = {}
            for a in cfr_actions:
                if a == taken:
                    action_values[a] = reward
                elif a == 'fold':
                    # check があれば fold は常に非合理（同じ損失）
                    action_values[a] = reward if can_check else reward * fold_fraction
                else:
                    # 他のアクション（call/raise系）は現在の報酬の控えめな推定
                    action_values[a] = reward * 0.8

            self.cfr.update_regret(state_key, taken, action_values)

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

            # 反実仮想価値: 各アクションを「代わりに取った場合」の推定値
            can_check = 'check' in strategy
            call_amount_rec: int = record.get('call_amount', 0)
            pot_rec: int = record.get('pot', 1)
            action_values: dict[str, float] = {}
            for a, prob in strategy.items():
                if a == taken:
                    action_values[a] = reward
                elif a == 'fold':
                    if can_check:
                        # check可能な状況でのfoldは支配戦略 → 後悔を蓄積させない
                        action_values[a] = reward
                    else:
                        # フォールドの価値: コール額がポット全体に占める比率で割引
                        # (コール額が小さいほどfoldはほぼ中立、大きいほど損失が大きい)
                        total = pot_rec + call_amount_rec
                        fold_fraction = call_amount_rec / total if total > 0 else 0.3
                        action_values[a] = reward * fold_fraction
                else:
                    # call / raise は現在の報酬を 80% で割り引いた推定
                    action_values[a] = reward * 0.8

            self.cfr.update_regret(state_key, taken, action_values)

        self._action_history.clear()
        self.cfr.save(self._save_path)
