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
from gto_strategy import build_state_key, get_equity_bucket, get_street
from gto_cfr import SimpleMCCFR

_DEFAULT_SAVE_PATH = "gto_strategy.json"

# ──────────────────────────────────────────────
# GTO 的なベットサイジング
# ──────────────────────────────────────────────
# ポットに対する倍率の選択肢
_BET_RATIOS = [0.33, 0.50, 0.67, 1.00, 1.50]

# equity bucket ごとのサイズ選択確率
# 強い手: オーバーベット寄り　弱い手（ブラフ）: 小さめ
_BET_SIZE_WEIGHTS: dict[int, list[float]] = {
    7: [0.05, 0.10, 0.25, 0.40, 0.20],   # ナッツ: 大きめ
    6: [0.10, 0.15, 0.35, 0.35, 0.05],
    5: [0.15, 0.25, 0.40, 0.18, 0.02],
    4: [0.25, 0.35, 0.30, 0.10, 0.00],
    3: [0.40, 0.35, 0.20, 0.05, 0.00],   # セミブラフ: 小さめ
    2: [0.55, 0.35, 0.10, 0.00, 0.00],
    1: [0.65, 0.30, 0.05, 0.00, 0.00],
    0: [0.70, 0.25, 0.05, 0.00, 0.00],   # ブラフ: 小さめ
}


def _pick_bet_size(eq_bucket: int, pot: int, min_raise: int) -> int:
    """確率的に GTO 的なベットサイズを選択する（10 刻みに丸める）"""
    weights = _BET_SIZE_WEIGHTS[eq_bucket]
    chosen_ratio = random.choices(_BET_RATIOS, weights=weights)[0]
    raw = int(pot * chosen_ratio)
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
    ) -> None:
        super().__init__(name, chips)
        self._save_path = save_path
        self._num_simulations = num_simulations
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
        state_key = build_state_key(equity, board, call_amount, pot, num_opponents)

        # ── 混合戦略を取得（CFR or ヒューリスティック）──
        strategy = self.cfr.get_strategy(state_key, valid_actions, eq_bucket)

        # ── 確率的にアクションを選択（GTOの核心）──
        chosen_action = random.choices(
            list(strategy.keys()), weights=list(strategy.values())
        )[0]

        # ── ベットサイズ決定 ──
        amount = 0
        if chosen_action == 'call':
            amount = call_amount
        elif chosen_action == 'raise':
            amount = _pick_bet_size(eq_bucket, pot, min_raise)

        # 行動履歴を保存（CFR 更新用）
        self._action_history.append({
            'state_key': state_key,
            'eq_bucket': eq_bucket,
            'strategy': strategy,
            'action': chosen_action,
        })

        return chosen_action, amount

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
            action_values: dict[str, float] = {}
            for a, prob in strategy.items():
                if a == taken:
                    action_values[a] = reward
                elif a == 'fold':
                    # フォールドは損失が確定（チップを投じた分だけ失う）
                    action_values[a] = reward * 0.3
                else:
                    # call / raise は現在の報酬を 80% で割り引いた推定
                    action_values[a] = reward * 0.8

            self.cfr.update_regret(state_key, taken, action_values)

        self._action_history.clear()
        self.cfr.save(self._save_path)
