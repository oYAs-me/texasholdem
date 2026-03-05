"""
gto_cfr.py

Regret Matching ベースの簡易 Monte Carlo CFR 実装。
未学習状態では gto_strategy.heuristic_strategy にフォールバックし、
ゲームを重ねるごとに戦略が洗練されていく。
"""
from __future__ import annotations
import json
import os
import tempfile
import warnings
from collections import defaultdict

from gto_strategy import heuristic_strategy

# JSONフォーマットのバージョン。状態キー形式が変わるたびに増やす
_FORMAT_VERSION = 4


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
    # この訪問回数を超えたら平均戦略（strategy_sum）でプレイする
    # CFR 理論では「現在の戦略」ではなく「全イテレーションの平均戦略」がナッシュ均衡に収束する
    _AVG_STRATEGY_THRESHOLD = 100

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
        num_players: int = 2,
        call_amount: int = 0,
        pot: int = 1,
        street: str = 'postflop',
        hand_potential: str = 'na',
    ) -> dict[str, float]:
        """
        現在の混合戦略を返す。
        訪問回数が閾値未満のうちはヒューリスティックにフォールバック。
        strategy_sum の蓄積は LEARN_THRESHOLD 到達後のみ（ヒューリスティック期間を除外）。
        """
        self.visit_count[state_key] += 1

        if self.visit_count[state_key] < self._LEARN_THRESHOLD:
            return heuristic_strategy(
                eq_bucket, valid_actions,
                num_players=num_players, state_key=state_key,
                call_amount=call_amount, pot=pot,
                street=street, hand_potential=hand_potential,
            )

        # Regret Matching: 正の後悔のみを使用して現在戦略を計算
        regrets = self.regret_sum[state_key]
        positive = {a: max(regrets.get(a, 0.0), 0.0) for a in valid_actions}
        total = sum(positive.values())

        if total > 0:
            current_strategy = {a: positive[a] / total for a in valid_actions}
        else:
            current_strategy = heuristic_strategy(
                eq_bucket, valid_actions,
                num_players=num_players, state_key=state_key,
                call_amount=call_amount, pot=pot,
                street=street, hand_potential=hand_potential,
            )

        # 平均戦略の累積（LEARN_THRESHOLD 以降のみ: ヒューリスティック期間を含まない）
        for a in valid_actions:
            self.strategy_sum[state_key][a] += current_strategy[a]

        # 十分学習された状態では平均戦略でプレイ（CFR 理論上の均衡戦略）
        if self.visit_count[state_key] >= self._AVG_STRATEGY_THRESHOLD:
            avg = self.strategy_sum[state_key]
            avg_total = sum(avg.get(a, 0.0) for a in valid_actions)
            if avg_total > 0:
                return {a: avg.get(a, 0.0) / avg_total for a in valid_actions}

        return current_strategy

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
            # CFR+: 負の累積後悔を 0 にリセット（収束速度を向上）
            if self.regret_sum[state_key][action] < 0:
                self.regret_sum[state_key][action] = 0.0

    def update_strategy_sum(self, state_key: str, strategy: dict[str, float]) -> None:
        """
        strategy_sum を直接更新する（realtime resolve 等から呼ぶ用）。
        LEARN_THRESHOLD 以降に達した状態のみ蓄積する。
        """
        if self.visit_count.get(state_key, 0) >= self._LEARN_THRESHOLD:
            for a, prob in strategy.items():
                self.strategy_sum[state_key][a] += prob

    # ──────────────────────────────────────────────
    # 永続化
    # ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        if path == os.devnull:
            return

        def _trim(d: dict) -> dict:
            return {k: round(v, 3) for k, v in d.items()}

        data = {
            '_version': _FORMAT_VERSION,
            'regret_sum':   {k: _trim(dict(v)) for k, v in self.regret_sum.items()},
            'strategy_sum': {k: _trim(dict(v)) for k, v in self.strategy_sum.items()},
            'visit_count':  dict(self.visit_count),
        }

        dir_name = os.path.dirname(os.path.abspath(path))
        try:
            with tempfile.NamedTemporaryFile(
                'w', dir=dir_name, suffix='.tmp', delete=False, encoding='utf-8'
            ) as f:
                tmp_path = f.name
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_version = data.get('_version', 0)
            if file_version != _FORMAT_VERSION:
                warnings.warn(
                    f"gto_strategy.json のバージョンが異なります "
                    f"(ファイル: v{file_version}, 現在: v{_FORMAT_VERSION})。"
                    "学習データをリセットして新規開始します。",
                    UserWarning,
                    stacklevel=2,
                )
                return

            for k, v in data.get('regret_sum', {}).items():
                self.regret_sum[k] = defaultdict(float, {ak: float(av) for ak, av in v.items()})
            for k, v in data.get('strategy_sum', {}).items():
                self.strategy_sum[k] = defaultdict(float, {ak: float(av) for ak, av in v.items()})
            for k, v in data.get('visit_count', {}).items():
                self.visit_count[k] = int(v)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            warnings.warn(
                f"gto_strategy.json の読み込みに失敗しました ({e})。"
                "学習データをリセットして新規開始します。",
                UserWarning,
                stacklevel=2,
            )

