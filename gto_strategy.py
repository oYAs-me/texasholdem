"""
gto_strategy.py

ゲーム状態の抽象化と、GTOヒューリスティック初期戦略テーブル。
CFR学習前でも合理的な混合戦略を提供する。
"""
from __future__ import annotations
from collections import Counter
from card import Board


# ──────────────────────────────────────────────
# ゲーム状態の抽象化
# ──────────────────────────────────────────────

def get_street(board: Board) -> str:
    if board.flops is None:
        return 'preflop'
    if board.turn is None:
        return 'flop'
    if board.river is None:
        return 'turn'
    return 'river'


def classify_board_texture(board: Board) -> str:
    """ボードのテクスチャを dry / semi_wet / wet に分類"""
    if board.flops is None:
        return 'dry'

    cards = list(board.flops)
    if board.turn:
        cards.append(board.turn)
    if board.river:
        cards.append(board.river)

    suits = [c.suit for c in cards]
    rank_set = {c.rank_int for c in cards}
    # A-5 Wheel: Ace を 1（ローエース）としても扱う
    if 14 in rank_set:
        rank_set.add(1)
    ranks = sorted(rank_set, reverse=True)

    max_suit_count = max(Counter(suits).values())

    # 連続ランクの最大長を計算
    max_consecutive = current = 1
    for i in range(len(ranks) - 1):
        if ranks[i] - ranks[i + 1] == 1:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 1

    if max_suit_count >= 3 or max_consecutive >= 3:
        return 'wet'
    if max_suit_count == 2 or max_consecutive == 2:
        return 'semi_wet'
    return 'dry'


def get_equity_bucket(equity: float) -> int:
    """エクイティを 0〜7 の 8 段階バケットに変換（0: <12.5%, 7: >87.5%）"""
    return min(int(equity * 8), 7)


def get_pot_odds_bucket(call_amount: int, pot: int) -> int:
    """ポットオッズを 0〜4 の 5 段階に変換。
    bucket 0 は call_amount==0（チェック可能）の専用バケット。
    コールが必要な場合は 1〜4 に割り当て、check状況と混同しない。
    """
    if call_amount == 0:
        return 0  # check専用
    total = pot + call_amount
    if total == 0:
        return 1
    return min(max(int(call_amount / total * 5), 1), 4)


def get_spr_bucket(chips: int, pot: int) -> int:
    """SPR (Stack-to-Pot Ratio) を 0〜2 の 3 段階に変換
    0: <1 (ショートスタック・コミット済み)
    1: 1〜4 (ミディアム)
    2: >4  (ディープスタック)
    """
    if pot <= 0:
        return 2
    spr = chips / pot
    if spr < 1:
        return 0
    if spr < 4:
        return 1
    return 2


def build_state_key(
    equity: float,
    board: Board,
    call_amount: int,
    pot: int,
    num_opponents: int,
    is_last_to_act: bool = False,
    chips: int = 1000,
    hand_potential: str = 'na',
) -> str:
    """CFR のルックアップキーを生成する"""
    street = get_street(board)
    eq_bucket = get_equity_bucket(equity)
    texture = classify_board_texture(board)
    po_bucket = get_pot_odds_bucket(call_amount, pot)
    opp_count = min(num_opponents, 4)
    pos = 'IP' if is_last_to_act else 'OOP'
    spr_bucket = get_spr_bucket(chips, pot)
    return f"{street}:{eq_bucket}:{texture}:{po_bucket}:{opp_count}:{pos}:{spr_bucket}:{hand_potential}"


# ──────────────────────────────────────────────
# ヒューリスティック初期戦略テーブル（GTOの近似）
# ──────────────────────────────────────────────
#
# GTO の原則:
#   - 強い手はバリューベット + 一定割合チェック（チェックレイズ用）
#   - 弱い手もブラフ頻度を維持（相手に自由なチェックを与えない）
#   - コール頻度はポットオッズで均衡させる（相手のブラフを無効化）
#   - 非常に弱い手（eq_bucket 0）もブラフとして一定確率でレイズ
#
# equity bucket 対応:
#   7: >87.5%   6: 75-87%   5: 62-75%   4: 50-62%
#   3: 37-50%   2: 25-37%   1: 12-25%   0: <12.5%

# ベット/チェック状況（call_amount == 0）
# raise_33 / raise_67 / raise_100 = ポットの 33% / 67% / 100% ベット
_BET_STRAT: dict[int, dict[str, float]] = {
    7: {'raise_33': 0.10, 'raise_67': 0.35, 'raise_100': 0.37, 'check': 0.18},  # ナッツ: 大きめ＋チェックレイズ準備
    6: {'raise_33': 0.15, 'raise_67': 0.35, 'raise_100': 0.20, 'check': 0.30},
    5: {'raise_33': 0.20, 'raise_67': 0.20, 'raise_100': 0.10, 'check': 0.50},
    4: {'raise_33': 0.20, 'raise_67': 0.07, 'raise_100': 0.03, 'check': 0.70},
    3: {'raise_33': 0.15, 'raise_67': 0.03, 'raise_100': 0.00, 'check': 0.82},  # セミブラフ: 小さめ
    2: {'raise_33': 0.10, 'raise_67': 0.02, 'raise_100': 0.00, 'check': 0.88},
    1: {'raise_33': 0.05, 'raise_67': 0.01, 'raise_100': 0.00, 'check': 0.94},
    0: {'raise_33': 0.12, 'raise_67': 0.02, 'raise_100': 0.00, 'check': 0.86},  # 弱手ブラフ: 小さめ
}

# コール/フォールド/レイズ状況（call_amount > 0）
_CALL_STRAT: dict[int, dict[str, float]] = {
    7: {'raise_33': 0.05, 'raise_67': 0.25, 'raise_100': 0.42, 'call': 0.28, 'fold': 0.00},
    6: {'raise_33': 0.10, 'raise_67': 0.20, 'raise_100': 0.12, 'call': 0.55, 'fold': 0.03},
    5: {'raise_33': 0.08, 'raise_67': 0.05, 'raise_100': 0.01, 'call': 0.80, 'fold': 0.06},
    4: {'raise_33': 0.03, 'raise_67': 0.01, 'raise_100': 0.00, 'call': 0.66, 'fold': 0.30},
    3: {'raise_33': 0.00, 'raise_67': 0.00, 'raise_100': 0.00, 'call': 0.42, 'fold': 0.58},
    2: {'raise_33': 0.00, 'raise_67': 0.00, 'raise_100': 0.00, 'call': 0.18, 'fold': 0.82},
    1: {'raise_33': 0.00, 'raise_67': 0.00, 'raise_100': 0.00, 'call': 0.07, 'fold': 0.93},
    0: {'raise_33': 0.00, 'raise_67': 0.00, 'raise_100': 0.00, 'call': 0.04, 'fold': 0.96},
}


def heuristic_strategy(eq_bucket: int, valid_actions: list[str]) -> dict[str, float]:
    """
    有効アクションに合わせた初期ヒューリスティック戦略を返す。
    CFR 未学習状態でのフォールバックとして使用する。
    """
    facing_bet = 'call' in valid_actions
    base = (_CALL_STRAT if facing_bet else _BET_STRAT)[eq_bucket].copy()

    # valid_actions にないものを除去して正規化
    filtered = {a: base.get(a, 0.0) for a in valid_actions}
    total = sum(filtered.values())
    if total == 0:
        n = len(valid_actions)
        return {a: 1.0 / n for a in valid_actions}
    return {a: v / total for a, v in filtered.items()}
