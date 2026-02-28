"""
gto_strategy.py

GTO 戦略のための状態抽象化とヒューリスティック。
状態空間（バケット）を細分化し、より高度な意思決定を可能にする。
"""
from __future__ import annotations
from typing import Any
from card import Board
from hand_strength import evaluate_hand, EvaluatedHand

# ──────────────────────────────────────────────
# 状態の抽象化 (Bucketing)
# ──────────────────────────────────────────────

def get_equity_bucket(equity: float) -> int:
    """エクイティを 0-19 の 20 段階に分類 (5% 刻み)"""
    return min(int(equity * 20), 19)

def get_street(board: Board) -> str:
    all_cards = board.get_all_cards()
    n = len(all_cards)
    if n == 0: return 'preflop'
    if n == 3: return 'flop'
    if n == 4: return 'turn'
    return 'river'

def classify_board_texture(board: Board) -> str:
    """
    ボードのテクスチャを分類する（レア状態のサンプリング用）。
    分類例: monotone, flush_draw, rainbow, paired, trips
    """
    cards = board.get_all_cards()
    if not cards: return 'na'
    
    suits = [c.suit for c in cards]
    ranks = [c.rank_int for c in cards]
    suit_counts = {s: suits.count(s) for s in set(suits)}
    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    
    max_suit = max(suit_counts.values()) if suit_counts else 0
    max_rank = max(rank_counts.values()) if rank_counts else 0
    
    if max_rank >= 3: return 'trips'
    if max_rank == 2: return 'paired'
    if max_suit >= 3: return 'monotone'
    if max_suit == 2: return 'flush_draw'
    return 'rainbow'

def get_hand_potential(hand_type: str, hand: Any, board: Board) -> str:
    """
    ハンドの強さと発展性を詳細に分類する。
    分類: nuts, strong, mid, weak_made, strong_draw, weak_draw, nothing
    """
    street = get_street(board)
    if street == 'preflop':
        return 'na'
    
    # 役の種類による基本分類
    if hand_type in ('STRAIGHT_FLUSH', 'FOUR_OF_A_KIND', 'FULL_HOUSE'):
        return 'nuts'
    if hand_type in ('FLUSH', 'STRAIGHT', 'THREE_OF_A_KIND'):
        return 'strong'
    if hand_type == 'TWO_PAIR':
        return 'mid'
    
    # ワンペアの場合の判定
    if hand_type == 'ONE_PAIR':
        # ボードの最大ランクと比較してトップペア以上か判定（簡易的）
        board_cards = board.get_all_cards()
        max_board_rank = max([c.rank_int for c in board_cards]) if board_cards else 0
        # 手札のペアのランクを取得
        from collections import Counter
        all_ranks = [c.rank_int for c in list(hand.cards) + list(board_cards)]
        counts = Counter(all_ranks)
        pair_rank = max([r for r, c in counts.items() if c >= 2])
        
        if pair_rank >= max_board_rank:
            return 'mid'  # トップペア以上
        return 'weak_made' # ミドル・ボトムペア
    
    # ハイカードの場合（ドローの判定）
    if street == 'river':
        return 'nothing' # リバーでハイカードは「無」
        
    # ドロー判定 (簡易実装: 本来は card.py 等で判定すべきだが、ここではエクイティと組み合わせて機能させる)
    # 実際には equity が高いハイカード = 強いドローとして機能する
    return 'draw' # 後続の処理で equity に応じて判断されるため、ここでは draw に統一

def build_state_key(
    equity: float,
    board: Board,
    call_amount: int,
    pot: int,
    num_opponents: int,
    is_last_to_act: bool = False,
    chips: int = 1000,
    hand_potential: str = 'na'
) -> str:
    """
    CFR 用の状態キー（InfoSet）を作成する。
    """
    street = get_street(board)
    texture = classify_board_texture(board)
    
    # ポットに対するコールの重み (0: check可, 1: 小, 2: 中, 3: 大)
    pot_ratio = 0
    if pot > 0:
        ratio = call_amount / pot
        if ratio == 0: pot_ratio = 0
        elif ratio <= 0.35: pot_ratio = 1
        elif ratio <= 0.75: pot_ratio = 2
        else: pot_ratio = 3
    
    # スタックの深さ (0: ショート, 1: 通常, 2: ディープ)
    stack_depth = 1
    if pot > 0:
        eff_stack = chips / pot
        if eff_stack < 3: stack_depth = 0
        elif eff_stack > 15: stack_depth = 2

    # 対戦人数 (1: heads-up, 2: multi-way)
    opp_count = 1 if num_opponents == 1 else 2

    # キーの組み立て
    # 例: "flop_paired_p2_s1_o2_pos1_mid"
    return f"{street}_{texture}_p{pot_ratio}_s{stack_depth}_o{opp_count}_pos{int(is_last_to_act)}_{hand_potential}"

# ──────────────────────────────────────────────
# ヒューリスティック戦略 (CFR の初期値・フォールバック)
# ──────────────────────────────────────────────

def heuristic_strategy(eq_bucket: int, valid_actions: list[str]) -> dict[str, float]:
    """
    エクイティに基づいた初期戦略を生成する。
    eq_bucket: 0-19 (5%刻み)
    """
    strategy = {a: 0.0 for a in valid_actions}
    equity = (eq_bucket + 0.5) / 20.0  # 中央値で計算
    
    # 非常に強い手 (Equity > 85%)
    if equity > 0.85:
        if 'raise_200' in valid_actions: strategy['raise_200'] = 0.2
        if 'raise_100' in valid_actions: strategy['raise_100'] = 0.3
        if 'raise_67' in valid_actions: strategy['raise_67'] = 0.2
        if 'raise_33' in valid_actions: strategy['raise_33'] = 0.1
        if 'call' in valid_actions: strategy['call'] = 0.2
        if 'check' in valid_actions: strategy['check'] = 0.2
    
    # 強い手 (Equity 70-85%)
    elif equity > 0.7:
        if 'raise_100' in valid_actions: strategy['raise_100'] = 0.2
        if 'raise_67' in valid_actions: strategy['raise_67'] = 0.3
        if 'raise_33' in valid_actions: strategy['raise_33'] = 0.2
        if 'call' in valid_actions: strategy['call'] = 0.2
        if 'check' in valid_actions: strategy['check'] = 0.3
    
    # 中堅の手 (Equity 45-70%)
    elif equity > 0.45:
        if 'raise_33' in valid_actions: strategy['raise_33'] = 0.1
        if 'call' in valid_actions: strategy['call'] = 0.6
        if 'check' in valid_actions: strategy['check'] = 0.7
        if 'fold' in valid_actions: strategy['fold'] = 0.1
    
    # 弱い手 (Equity 25-45%)
    elif equity > 0.25:
        if 'call' in valid_actions: strategy['call'] = 0.3
        if 'check' in valid_actions: strategy['check'] = 0.5
        if 'fold' in valid_actions: strategy['fold'] = 0.5
        # ブラフ: 低確率でレイズ
        if 'raise_67' in valid_actions: strategy['raise_67'] = 0.05
    
    # ほぼゴミ (Equity < 25%)
    else:
        if 'check' in valid_actions: strategy['check'] = 0.4
        if 'fold' in valid_actions: strategy['fold'] = 0.6
        # ブラフ: 極低確率で大きなレイズ
        if 'raise_100' in valid_actions: strategy['raise_100'] = 0.02

    # 合計を 1.0 に正規化
    total = sum(strategy.values())
    if total > 0:
        return {a: v / total for a, v in strategy.items()}
    else:
        # 万が一の場合は等確率
        return {a: 1.0/len(valid_actions) for a in valid_actions}
