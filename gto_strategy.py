"""
gto_strategy.py

GTO 戦略のための状態抽象化・ヒューリスティック・アクション価値計算。
ストリート（preflop/flop/turn/river）ごとに適切なロジックを適用する。
"""
from __future__ import annotations
from collections import Counter
from typing import Any
from card import Board
from hand_strength import evaluate_hand, EvaluatedHand

# ──────────────────────────────────────────────
# 状態の抽象化 (Bucketing)
# ──────────────────────────────────────────────

def get_equity_bucket(equity: float) -> int:
    """エクイティを 0-19 の 20 段階に分類 (5% 刻み)。ヒューリスティック用。"""
    return min(int(equity * 20), 19)

def get_equity_bucket_5(equity: float) -> int:
    """エクイティを 0-4 の 5 段階に分類 (20% 刻み)。状態キー用（収束速度優先）。"""
    return min(int(equity * 5), 4)

def get_street(board: Board) -> str:
    n = len(board.get_all_cards())
    if n == 0: return 'preflop'
    if n == 3: return 'flop'
    if n == 4: return 'turn'
    return 'river'

def classify_board_texture(board: Board) -> str:
    """ボードのテクスチャを分類する。Counter を使用して O(n) で処理。"""
    cards = board.get_all_cards()
    if not cards:
        return 'na'

    suit_counts = Counter(c.suit for c in cards)
    rank_counts = Counter(c.rank_int for c in cards)
    max_suit = max(suit_counts.values())
    max_rank = max(rank_counts.values())

    if max_rank >= 3: return 'trips'
    if max_rank == 2: return 'paired'
    if max_suit >= 3: return 'monotone'
    if max_suit == 2: return 'flush_draw'
    return 'rainbow'

def get_hand_potential(hand_type: str, hand: Any, board: Board) -> str:
    """
    ハンドの強さと発展性を分類する。
    戻り値: nuts / strong / mid / weak_made / draw / nothing / na
    """
    street = get_street(board)
    if street == 'preflop':
        return 'na'

    if hand_type in ('STRAIGHT_FLUSH', 'FOUR_OF_A_KIND', 'FULL_HOUSE'):
        return 'nuts'
    if hand_type in ('FLUSH', 'STRAIGHT', 'THREE_OF_A_KIND'):
        return 'strong'
    if hand_type == 'TWO_PAIR':
        return 'mid'

    if hand_type == 'ONE_PAIR':
        board_cards = board.get_all_cards()
        max_board_rank = max(c.rank_int for c in board_cards) if board_cards else 0
        all_ranks = [c.rank_int for c in list(hand.cards) + list(board_cards)]
        rank_counts = Counter(all_ranks)
        pair_rank = max(r for r, c in rank_counts.items() if c >= 2)
        if pair_rank >= max_board_rank:
            return 'mid'       # トップペア以上
        return 'weak_made'     # ミドル・ボトムペア

    if street == 'river':
        return 'nothing'
    return 'draw'              # フロップ・ターンでのハイカード = ドロー候補


# ──────────────────────────────────────────────
# 状態キー生成
# ──────────────────────────────────────────────

def build_state_key(
    equity: float,
    board: Board,
    call_amount: int,
    pot: int,
    num_opponents: int,
    is_last_to_act: bool = False,
    chips: int = 1000,
    hand_potential: str = 'na',
    hand_key: str | None = None,
    position: str | None = None,
    facing_action: str | None = None,
    my_round_bet: int = 0,
) -> str:
    """
    CFR 用の状態キー（InfoSet）を作成する。

    プリフロップ: hand_key / position / facing_action が揃えば 169ハンドキーを使用。
    ポストフロップ: ストリート・テクスチャ・エクイティ 5段階・ハンド強度 etc.
    """
    street = get_street(board)
    did_invest = 1 if my_round_bet > 20 else 0
    # 自分が既にチップを投資済み かつ コールが必要 = リレイズを受けている状態
    facing_reraise = 1 if (my_round_bet > 0 and call_amount > 0) else 0

    # ── プリフロップ ──
    if street == 'preflop' and hand_key is not None and position is not None and facing_action is not None:
        return f"preflop_{hand_key}_{position}_{facing_action}_i{did_invest}_fr{facing_reraise}"

    # ── ポストフロップ ──
    texture = classify_board_texture(board)
    eq5 = get_equity_bucket_5(equity)

    # ポットオッズバケット (0: check可能, 1: 小額コール, 2: 中額, 3: 大額)
    pot_ratio = 0
    if pot > 0 and call_amount > 0:
        ratio = call_amount / pot
        if ratio <= 0.35:   pot_ratio = 1
        elif ratio <= 0.75: pot_ratio = 2
        else:               pot_ratio = 3

    # スタック深さ (0: ショート/コミット, 1: 通常, 2: ディープ)
    stack_depth = 1
    if pot > 0:
        eff_stack = chips / pot
        if eff_stack < 3:   stack_depth = 0
        elif eff_stack > 15: stack_depth = 2

    opp_count = 1 if num_opponents == 1 else 2

    return (f"{street}_{texture}_p{pot_ratio}_s{stack_depth}"
            f"_o{opp_count}_pos{int(is_last_to_act)}_eq{eq5}_{hand_potential}"
            f"_i{did_invest}_fr{facing_reraise}")


# ──────────────────────────────────────────────
# アクション価値計算（CFR 学習用）
# ──────────────────────────────────────────────

def compute_action_values(
    reward: float,
    equity: float,
    taken_action: str,
    all_actions: list[str],
    call_amount: int,
    pot: int,
    street: str = 'postflop',
    hand_potential: str = 'na',
) -> dict[str, float]:
    """
    各アクションの反実仮想価値（counterfactual value）をポットオッズ理論から推定する。

    - fold  : チップを節約する行動。エクイティとポットオッズの関係で価値が決まる。
    - check : 無料でターンを見る。エクイティと同等の期待値。
    - call  : ポットオッズに対してエクイティが見合うか。
    - raise : 強いハンドほど追加価値が大きい。弱いハンドはリスクが高い。
    """
    pot_odds = (call_amount / (pot + call_amount)
                if (pot + call_amount) > 0 and call_amount > 0 else 0.0)
    can_check = 'check' in all_actions
    is_draw = hand_potential == 'draw'

    values: dict[str, float] = {}
    for a in all_actions:
        if a == taken_action:
            values[a] = reward

        elif a == 'fold':
            if can_check:
                # check が選択肢にある時の fold は支配される戦略 → 同じ結果
                values[a] = reward
            elif equity > pot_odds + 0.05:
                # +EV なのに fold は明らかな損失 → 強い負の後悔を付与
                values[a] = reward * 0.2
            else:
                # 呼び込みが-EV: fold は適切な損切り（中立値 0）
                values[a] = 0.0

        elif a == 'check':
            # 無料でターンを見る: 実際の結果と概ね同等
            values[a] = reward

        elif a == 'call':
            if call_amount <= 0:
                values[a] = reward  # 実質チェック
            elif equity >= pot_odds:
                # +EV コール: 実際の結果と同等
                values[a] = reward
            else:
                # -EV コール: ポットオッズに対するショートフォール分割引
                values[a] = reward * max(equity / max(pot_odds, 0.01), 0.1)

        else:
            # raise_33 / raise_67 / raise_100 / raise_200
            if hand_potential in ('nuts', 'strong'):
                # ナッツ級・強いハンド: レイズで価値を最大化
                values[a] = reward
            elif hand_potential == 'mid' or (is_draw and street in ('flop', 'turn')):
                # 中程度ハンド・ドロー: セミブラフ価値あり
                values[a] = reward * 0.7
            elif street == 'river':
                # リバーでの弱いハンドのレイズはほぼブラフのみ → 低評価
                values[a] = reward * max(2 * equity - 1.0, -0.5)
            else:
                # その他: equity に比例した中間評価
                values[a] = reward * (2 * equity - 0.5)

    return values


# ──────────────────────────────────────────────
# ヒューリスティック戦略 (CFR の初期値・フォールバック)
# ──────────────────────────────────────────────

def heuristic_strategy(
    eq_bucket: int,
    valid_actions: list[str],
    num_players: int = 2,
    state_key: str = '',
    call_amount: int = 0,
    pot: int = 1,
    street: str = 'postflop',
    hand_potential: str = 'na',
) -> dict[str, float]:
    """
    エクイティ・ストリート・ハンド強度に基づいた初期戦略を生成する。

    eq_bucket: 0-19 (5%刻み)
    street   : preflop / flop / turn / river (ストリート別の重みを適用)
    hand_potential: nuts / strong / mid / weak_made / draw / nothing / na
    """
    # プリフロップは 169ハンドテーブルに委譲
    if state_key.startswith('preflop_') or street == 'preflop':
        try:
            from preflop_gto import preflop_heuristic
            parts = state_key.split('_')
            if len(parts) >= 4:
                hand_key = parts[1]
                position  = parts[2]
                facing    = parts[3]
                return preflop_heuristic(hand_key, position, facing, valid_actions,
                                         call_amount=call_amount, pot=pot)
        except ImportError:
            pass  # preflop_gto が存在しない場合はポストフロップ戦略で代替

    strategy = {a: 0.0 for a in valid_actions}
    equity = (eq_bucket + 0.5) / 20.0  # 5%刻みバケットの中央値

    # 人数スケーリング: 多人数ではエクイティ閾値を下げる
    # 2人戦の閾値をベースに、人数が増えるほど閾値が低くなる
    scale = 2.0 / max(num_players, 2)
    pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 and call_amount > 0 else 0.0

    # リレイズ対応: 状態キーから facing_reraise フラグを読み取る
    facing_reraise = '_fr1' in state_key

    # ── リレイズ時の特別処理 ──
    # 相手のリレイズレンジは強いハンドに絞られる → こちらは大きく絞る
    if facing_reraise:
        return _reraise_strategy(strategy, equity, hand_potential, valid_actions, pot_odds)

    # ──────────────────────────
    # ストリート別ヒューリスティック
    # ──────────────────────────

    if street == 'river':
        # リバー: ドローなし。強いハンドは強くバリューベット。弱いハンドはチェック/フォールド。
        _river_strategy(strategy, equity, hand_potential, valid_actions, scale, pot_odds)

    elif street == 'turn':
        # ターン: メイドハンドは強くベット（ドロー否定）。ドローはポットオッズ次第。
        _turn_strategy(strategy, equity, hand_potential, valid_actions, scale, pot_odds)

    elif street == 'flop':
        # フロップ: Cベット重視。ドロー・セミブラフ価値あり。
        _flop_strategy(strategy, equity, hand_potential, valid_actions, scale, pot_odds)

    else:
        # ポストフロップ汎用（フォールバック）
        _generic_strategy(strategy, equity, valid_actions, scale)

    return _normalize(strategy, valid_actions)


def _reraise_strategy(
    strategy: dict, equity: float, hand_potential: str,
    valid_actions: list[str], pot_odds: float,
) -> dict[str, float]:
    """
    リレイズ対応ヒューリスティック。

    相手がリレイズしてきた = 相手のレンジは強い。
    → ナッツ級のみ4-bet、強いハンドはコール、それ以外はフォールド。
    レイズは最小限（ナッツ時のみ）。
    """
    if hand_potential in ('nuts',):
        # ナッツ: 積極的に4-bet
        if 'raise_100' in valid_actions: strategy['raise_100'] = 0.40
        if 'raise_67'  in valid_actions: strategy['raise_67']  = 0.30
        if 'raise_200' in valid_actions: strategy['raise_200'] = 0.20
        if 'call'  in valid_actions: strategy['call']  = 0.10

    elif hand_potential == 'strong' or equity >= 0.70:
        # 強いハンド: コール主体、稀に4-bet
        if 'raise_67' in valid_actions: strategy['raise_67'] = 0.10
        if 'raise_33' in valid_actions: strategy['raise_33'] = 0.10
        if 'call' in valid_actions: strategy['call'] = 0.60
        if 'fold' in valid_actions: strategy['fold'] = 0.20

    elif hand_potential == 'mid' and equity >= pot_odds + 0.10:
        # 中程度ハンドかつポットオッズが十分: コール、レイズ禁止
        if 'call'  in valid_actions: strategy['call']  = 0.55
        if 'check' in valid_actions: strategy['check'] = 0.20
        if 'fold'  in valid_actions: strategy['fold']  = 0.25

    elif equity >= pot_odds + 0.05:
        # ポットオッズが辛うじて合う: チェックかコールのみ
        if 'check' in valid_actions: strategy['check'] = 0.35
        if 'call'  in valid_actions: strategy['call']  = 0.35
        if 'fold'  in valid_actions: strategy['fold']  = 0.30

    else:
        # ポットオッズ不足 or 弱いハンド: 大半フォールド
        if 'check' in valid_actions: strategy['check'] = 0.30
        if 'fold'  in valid_actions: strategy['fold']  = 0.65
        if 'call'  in valid_actions: strategy['call']  = 0.05

    return _normalize(strategy, valid_actions)


def _river_strategy(
    strategy: dict, equity: float, hand_potential: str,
    valid_actions: list[str], scale: float, pot_odds: float,
) -> None:
    """
    リバーヒューリスティック。
    ドローが完成しないため、ショーダウン価値が全て。

    - nuts/strong: 大きくバリューベット
    - mid       : ミディアムベット
    - weak_made : ポットオッズ次第でコール/チェック
    - nothing   : チェック/フォールド
    """
    th_vs = 0.80 * scale
    th_s  = 0.60 * scale
    th_m  = 0.40 * scale

    if hand_potential in ('nuts', 'strong') or equity > th_vs:
        if 'raise_100' in valid_actions: strategy['raise_100'] = 0.35
        if 'raise_67'  in valid_actions: strategy['raise_67']  = 0.30
        if 'raise_200' in valid_actions: strategy['raise_200'] = 0.15
        if 'call'  in valid_actions: strategy['call']  = 0.15
        if 'check' in valid_actions: strategy['check'] = 0.05

    elif hand_potential == 'mid' or equity > th_s:
        if 'raise_67'  in valid_actions: strategy['raise_67']  = 0.25
        if 'raise_33'  in valid_actions: strategy['raise_33']  = 0.20
        if 'call'  in valid_actions: strategy['call']  = 0.35
        if 'check' in valid_actions: strategy['check'] = 0.20

    elif hand_potential == 'weak_made' or equity > th_m:
        if 'call'  in valid_actions: strategy['call']  = 0.50
        if 'check' in valid_actions: strategy['check'] = 0.40
        if 'fold'  in valid_actions: strategy['fold']  = 0.10

    else:
        # nothing / draw失敗: チェック or フォールド（ほぼブラフなし）
        if 'check' in valid_actions: strategy['check'] = 0.70
        if 'fold'  in valid_actions: strategy['fold']  = 0.30


def _turn_strategy(
    strategy: dict, equity: float, hand_potential: str,
    valid_actions: list[str], scale: float, pot_odds: float,
) -> None:
    """
    ターンヒューリスティック。
    メイドハンドはドロー否定のために積極的にベット。
    ドロー（draw）はポットオッズ次第でコール or セミブラフ。
    """
    th_vs = 0.75 * scale
    th_s  = 0.55 * scale
    th_m  = 0.38 * scale

    if hand_potential in ('nuts', 'strong') or equity > th_vs:
        # 強いハンド: 積極的にベットしてドローを除去
        if 'raise_100' in valid_actions: strategy['raise_100'] = 0.30
        if 'raise_67'  in valid_actions: strategy['raise_67']  = 0.30
        if 'raise_200' in valid_actions: strategy['raise_200'] = 0.10
        if 'call'  in valid_actions: strategy['call']  = 0.20
        if 'check' in valid_actions: strategy['check'] = 0.10

    elif hand_potential == 'mid' or equity > th_s:
        if 'raise_67'  in valid_actions: strategy['raise_67']  = 0.20
        if 'raise_33'  in valid_actions: strategy['raise_33']  = 0.20
        if 'call'  in valid_actions: strategy['call']  = 0.40
        if 'check' in valid_actions: strategy['check'] = 0.20

    elif hand_potential == 'draw':
        # ドロー: ポットオッズとセミブラフ価値を考慮
        if equity > pot_odds:
            if 'call'  in valid_actions: strategy['call']  = 0.50
            if 'raise_33' in valid_actions: strategy['raise_33'] = 0.20  # セミブラフ
            if 'check' in valid_actions: strategy['check'] = 0.20
            if 'fold'  in valid_actions: strategy['fold']  = 0.10
        else:
            if 'check' in valid_actions: strategy['check'] = 0.50
            if 'fold'  in valid_actions: strategy['fold']  = 0.40
            if 'call'  in valid_actions: strategy['call']  = 0.10

    elif hand_potential == 'weak_made' or equity > th_m:
        if 'call'  in valid_actions: strategy['call']  = 0.45
        if 'check' in valid_actions: strategy['check'] = 0.40
        if 'fold'  in valid_actions: strategy['fold']  = 0.15

    else:
        if 'check' in valid_actions: strategy['check'] = 0.45
        if 'fold'  in valid_actions: strategy['fold']  = 0.45
        if 'call'  in valid_actions: strategy['call']  = 0.10


def _flop_strategy(
    strategy: dict, equity: float, hand_potential: str,
    valid_actions: list[str], scale: float, pot_odds: float,
) -> None:
    """
    フロップヒューリスティック。
    Cベット（コンティニュエーションベット）重視。
    ドローやセミブラフを多く含む。
    """
    th_vs = 0.75 * scale
    th_s  = 0.55 * scale
    th_m  = 0.35 * scale
    th_w  = 0.20 * scale

    if hand_potential in ('nuts', 'strong') or equity > th_vs:
        if 'raise_100' in valid_actions: strategy['raise_100'] = 0.20
        if 'raise_67'  in valid_actions: strategy['raise_67']  = 0.30
        if 'raise_33'  in valid_actions: strategy['raise_33']  = 0.15
        if 'call'  in valid_actions: strategy['call']  = 0.20
        if 'check' in valid_actions: strategy['check'] = 0.15

    elif hand_potential == 'mid' or equity > th_s:
        if 'raise_67'  in valid_actions: strategy['raise_67']  = 0.15
        if 'raise_33'  in valid_actions: strategy['raise_33']  = 0.20
        if 'call'  in valid_actions: strategy['call']  = 0.40
        if 'check' in valid_actions: strategy['check'] = 0.25

    elif hand_potential == 'draw' or equity > th_m:
        # セミブラフ: フロップは多くのドローが存在
        if 'raise_33'  in valid_actions: strategy['raise_33']  = 0.15
        if 'call'  in valid_actions: strategy['call']  = 0.45
        if 'check' in valid_actions: strategy['check'] = 0.30
        if 'fold'  in valid_actions: strategy['fold']  = 0.10

    elif equity > th_w:
        if 'call'  in valid_actions: strategy['call']  = 0.35
        if 'check' in valid_actions: strategy['check'] = 0.45
        if 'fold'  in valid_actions: strategy['fold']  = 0.20

    else:
        if 'check' in valid_actions: strategy['check'] = 0.45
        if 'fold'  in valid_actions: strategy['fold']  = 0.50
        if 'call'  in valid_actions: strategy['call']  = 0.05


def _generic_strategy(
    strategy: dict, equity: float, valid_actions: list[str], scale: float,
) -> None:
    """汎用ポストフロップヒューリスティック（フォールバック）。"""
    th_vs = 0.85 * scale
    th_s  = 0.70 * scale
    th_m  = 0.45 * scale
    th_w  = 0.25 * scale

    if equity > th_vs:
        if 'raise_200' in valid_actions: strategy['raise_200'] = 0.20
        if 'raise_100' in valid_actions: strategy['raise_100'] = 0.30
        if 'raise_67'  in valid_actions: strategy['raise_67']  = 0.20
        if 'raise_33'  in valid_actions: strategy['raise_33']  = 0.10
        if 'call'  in valid_actions: strategy['call']  = 0.20
        if 'check' in valid_actions: strategy['check'] = 0.20
    elif equity > th_s:
        if 'raise_100' in valid_actions: strategy['raise_100'] = 0.20
        if 'raise_67'  in valid_actions: strategy['raise_67']  = 0.30
        if 'raise_33'  in valid_actions: strategy['raise_33']  = 0.20
        if 'call'  in valid_actions: strategy['call']  = 0.20
        if 'check' in valid_actions: strategy['check'] = 0.30
    elif equity > th_m:
        if 'raise_33' in valid_actions: strategy['raise_33'] = 0.10
        if 'call'  in valid_actions: strategy['call']  = 0.60
        if 'check' in valid_actions: strategy['check'] = 0.70
        if 'fold'  in valid_actions: strategy['fold']  = 0.10
    elif equity > th_w:
        if 'call'  in valid_actions: strategy['call']  = 0.30
        if 'check' in valid_actions: strategy['check'] = 0.50
        if 'fold'  in valid_actions: strategy['fold']  = 0.50
        if 'raise_67' in valid_actions: strategy['raise_67'] = 0.05
    else:
        if 'check' in valid_actions: strategy['check'] = 0.40
        if 'fold'  in valid_actions: strategy['fold']  = 0.60
        if 'raise_100' in valid_actions: strategy['raise_100'] = 0.02


def _normalize(strategy: dict[str, float], valid_actions: list[str]) -> dict[str, float]:
    """戦略確率を合計1に正規化する。"""
    total = sum(strategy.values())
    if total > 0:
        return {a: strategy[a] / total for a in valid_actions}
    return {a: 1.0 / len(valid_actions) for a in valid_actions}


