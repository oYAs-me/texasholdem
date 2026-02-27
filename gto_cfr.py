"""
gto_cfr.py

Regret Matching ベースの簡易 Monte Carlo CFR 実装。
未学習状態では gto_strategy.heuristic_strategy にフォールバックし、
ゲームを重ねるごとに戦略が洗練されていく。
"""
from __future__ import annotations
import json
import os
from collections import defaultdict

from gto_strategy import heuristic_strategy


class SimpleMCCFR:
    """
    各ゲーム状態（state_key）ごとに累積後悔（regret）と
    累積戦略（strategy）を管理する Regret Matching エンジン。

    学習サイクル:
        1. get_strategy() でアクション確率を取得
        2. アクションを実行してゲーム結果を観測
        3. update_regret() で後悔を更新
        4. save() で JSON に書き出し（次セッションで引き継ぎ）
    """

    # この訪問回数を超えたら CFR 戦略を使う（それ以下はヒューリスティック）
    _LEARN_THRESHOLD = 20

    def __init__(self) -> None:
        # {state_key: {action: 累積後悔}}
        self.regret_sum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # {state_key: {action: 累積戦略確率}}
        self.strategy_sum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # {state_key: 訪問回数}
        self.visit_count: dict[str, int] = defaultdict(int)

    # ──────────────────────────────────────────────
    # 戦略の取得
    # ──────────────────────────────────────────────

    def get_strategy(
        self,
        state_key: str,
        valid_actions: list[str],
        eq_bucket: int,
    ) -> dict[str, float]:
        """
        現在の混合戦略を返す。
        訪問回数が閾値未満のうちはヒューリスティックにフォールバック。
        """
        self.visit_count[state_key] += 1

        if self.visit_count[state_key] < self._LEARN_THRESHOLD:
            return heuristic_strategy(eq_bucket, valid_actions)

        # Regret Matching: 正の後悔のみを使用して確率を計算
        regrets = self.regret_sum[state_key]
        positive = {a: max(regrets.get(a, 0.0), 0.0) for a in valid_actions}
        total = sum(positive.values())

        if total > 0:
            strategy = {a: positive[a] / total for a in valid_actions}
        else:
            strategy = heuristic_strategy(eq_bucket, valid_actions)

        # 平均戦略の累積（収束後に average strategy を使えるようにする）
        for a in valid_actions:
            self.strategy_sum[state_key][a] += strategy[a]

        return strategy

    # ──────────────────────────────────────────────
    # 学習（後悔の更新）
    # ──────────────────────────────────────────────

    def update_regret(
        self,
        state_key: str,
        taken_action: str,
        action_values: dict[str, float],
    ) -> None:
        """
        反実仮想後悔（counterfactual regret）を更新する。

        action_values: {action: 推定価値}
            取ったアクションより良かったアクションは正の後悔、
            悪かったアクションは負の後悔として蓄積する。
        """
        taken_value = action_values.get(taken_action, 0.0)
        for action, value in action_values.items():
            self.regret_sum[state_key][action] += value - taken_value

    # ──────────────────────────────────────────────
    # 永続化
    # ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            'regret_sum': {k: dict(v) for k, v in self.regret_sum.items()},
            'strategy_sum': {k: dict(v) for k, v in self.strategy_sum.items()},
            'visit_count': dict(self.visit_count),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in data.get('regret_sum', {}).items():
                self.regret_sum[k] = defaultdict(float, {ak: float(av) for ak, av in v.items()})
            for k, v in data.get('strategy_sum', {}).items():
                self.strategy_sum[k] = defaultdict(float, {ak: float(av) for ak, av in v.items()})
            for k, v in data.get('visit_count', {}).items():
                self.visit_count[k] = int(v)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
