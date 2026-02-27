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
    ranks = sorted({c.rank_int for c in cards}, reverse=True)

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
    """ポットオッズを 0〜4 の 5 段階に変換"""
    total = pot + call_amount
    if total == 0:
        return 0
    return min(int(call_amount / total * 5), 4)


def build_state_key(
    equity: float,
    board: Board,
    call_amount: int,
    pot: int,
    num_opponents: int,
) -> str:
    """CFR のルックアップキーを生成する"""
    street = get_street(board)
    eq_bucket = get_equity_bucket(equity)
    texture = classify_board_texture(board)
    po_bucket = get_pot_odds_bucket(call_amount, pot)
    opp_count = min(num_opponents, 4)
    return f"{street}:{eq_bucket}:{texture}:{po_bucket}:{opp_count}"


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
_BET_STRAT: dict[int, dict[str, float]] = {
    7: {'raise': 0.82, 'check': 0.18},   # ナッツ: ほぼベット、一部チェックレイズ準備
    6: {'raise': 0.70, 'check': 0.30},
    5: {'raise': 0.50, 'check': 0.50},   # 強い手を均等に混合
    4: {'raise': 0.30, 'check': 0.70},
    3: {'raise': 0.18, 'check': 0.82},   # セミブラフ頻度
    2: {'raise': 0.12, 'check': 0.88},
    1: {'raise': 0.06, 'check': 0.94},
    0: {'raise': 0.14, 'check': 0.86},   # 弱手のブラフ（均衡維持）
}

# コール/フォールド/レイズ状況（call_amount > 0）
_CALL_STRAT: dict[int, dict[str, float]] = {
    7: {'raise': 0.72, 'call': 0.28, 'fold': 0.00},
    6: {'raise': 0.42, 'call': 0.55, 'fold': 0.03},
    5: {'raise': 0.14, 'call': 0.80, 'fold': 0.06},
    4: {'raise': 0.04, 'call': 0.66, 'fold': 0.30},
    3: {'raise': 0.00, 'call': 0.42, 'fold': 0.58},
    2: {'raise': 0.00, 'call': 0.18, 'fold': 0.82},
    1: {'raise': 0.00, 'call': 0.07, 'fold': 0.93},
    0: {'raise': 0.00, 'call': 0.04, 'fold': 0.96},
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
