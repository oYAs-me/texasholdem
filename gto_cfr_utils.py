"""
gto_cfr_utils.py

CFR データのマージ・ロード・保存に関する共通ユーティリティ。
gto_selfplay.py と gto_rare_training.py の重複コードを集約する。
"""
from __future__ import annotations

from collections import defaultdict

from gto_cfr import SimpleMCCFR


def merge_cfr_data(results: list[dict]) -> dict:
    """
    複数プロセスの CFR 結果を合算してマージする。
    CFR は独立したシミュレーションの regret を足し合わせることで
    等価な学習効果が得られる。
    """
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


def apply_merged_data(cfr: SimpleMCCFR, data: dict) -> None:
    """マージ済みデータを SimpleMCCFR インスタンスに書き込む"""
    for k, v in data["regret_sum"].items():
        cfr.regret_sum[k] = defaultdict(float, v)
    for k, v in data["strategy_sum"].items():
        cfr.strategy_sum[k] = defaultdict(float, v)
    for k, v in data["visit_count"].items():
        cfr.visit_count[k] = v


def load_base(save_path: str) -> dict:
    """既存の学習データをマージ用の dict として読み込む"""
    base_cfr = SimpleMCCFR()
    base_cfr.load(save_path)
    return {
        "regret_sum":   {k: dict(v) for k, v in base_cfr.regret_sum.items()},
        "strategy_sum": {k: dict(v) for k, v in base_cfr.strategy_sum.items()},
        "visit_count":  dict(base_cfr.visit_count),
    }


def save_merged(results: list[dict], save_path: str) -> int:
    """マージして保存。学習済み状態数を返す"""
    merged = merge_cfr_data(results)
    final_cfr = SimpleMCCFR()
    apply_merged_data(final_cfr, merged)
    final_cfr.save(save_path)
    return len(merged["visit_count"])


def cfr_to_dict(cfr: SimpleMCCFR) -> dict:
    """SimpleMCCFR インスタンスをマージ用 dict に変換する"""
    return {
        "regret_sum":   {k: dict(v) for k, v in cfr.regret_sum.items()},
        "strategy_sum": {k: dict(v) for k, v in cfr.strategy_sum.items()},
        "visit_count":  dict(cfr.visit_count),
    }
