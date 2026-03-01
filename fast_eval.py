"""
fast_eval.py

高速ポーカーハンド評価モジュール。

カード表現: card_int = (rank - 2) * 4 + suit
  rank: 2-14 → 0-12  (card_int // 4 + 2 でデコード)
  suit: 0='s', 1='h', 2='d', 3='c'  (card_int % 4 でデコード)

evaluate_hand() の Counter ベースの実装を整数配列索引で置き換え、
calculate_equity() を numpy 一括サンプリングに置き換えることで
高速化を実現する。

最適化のポイント:
- sorted() 呼び出しを最小化：rc を 14→2 の降順でスキャン
- Counter オブジェクト不使用（rc = [0]*15 リスト索引）
- _hand_score は定数倍算のみ（リスト生成なし）
- numpy 一括サンプリング（argsort による置換サンプリング）
"""
from __future__ import annotations

import numpy as np

# スートの文字 → 整数マッピング
_SUIT_TO_INT: dict[str, int] = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

# スコア計算定数（15進数エンコード）
_B6 = 15 ** 6  # hand_rank
_B5 = 15 ** 5  # primary
_B4 = 15 ** 4  # secondary
_B3 = 15 ** 3  # k0
_B2 = 15 ** 2  # k1
_B1 = 15       # k2
# k3 は × 1


def card_to_int(card) -> int:
    """Card オブジェクト → card_int"""
    return (card.rank_int - 2) * 4 + _SUIT_TO_INT[card.suit]


def _hs(hr: int, p: int = 0, s: int = 0,
         k0: int = 0, k1: int = 0, k2: int = 0, k3: int = 0) -> int:
    """ハンドスコア計算（引数展開で list 生成を回避）"""
    return hr * _B6 + p * _B5 + s * _B4 + k0 * _B3 + k1 * _B2 + k2 * _B1 + k3


def _straight_high_desc(unique_r: list[int]) -> int:
    """
    降順ユニークランクリストからストレートの最高位ランクを返す（なければ0）。
    unique_r は重複なし・降順であることが前提。
    """
    n = len(unique_r)
    for i in range(n - 4):
        if unique_r[i] - unique_r[i + 4] == 4:
            return unique_r[i]
    # A-2-3-4-5 ホイール
    if n >= 5 and unique_r[0] == 14 and unique_r[-1] <= 5:
        s = set(unique_r)
        if 2 in s and 3 in s and 4 in s and 5 in s:
            return 5
    return 0


def evaluate_7_score(cards7) -> int:
    """
    7枚のカード整数から手の強さスコアを返す高速評価関数。

    cards7: iterable of 7 int (card_int = (rank-2)*4 + suit)
    戻り値: int (大きいほど強い手)

    最適化:
    - max_cnt による早期ルーティング（不要なループをスキップ）
    - ビットマスクによるストレート判定（リスト不要）
    - sorted() はフラッシュのみ（最小限）
    - list.append ゼロ（ホットパスでリスト生成なし）
    """
    # ── デコード + カウント（1パス）──
    rc = [0] * 15
    sc = [0, 0, 0, 0]
    max_cnt = 0
    for c in cards7:
        r = c // 4 + 2
        s = c % 4
        rc[r] += 1
        sc[s] += 1
        if rc[r] > max_cnt:
            max_cnt = rc[r]

    # フラッシュ用にランク/スートのリストが必要な場合のみ展開
    flush_suit = -1
    for i in range(4):
        if sc[i] >= 5:
            flush_suit = i
            break

    if flush_suit >= 0:
        # フラッシュ確定 → ランク/スートのリストを展開
        fr = sorted([c // 4 + 2 for c in cards7 if c % 4 == flush_suit], reverse=True)
        sf = _straight_high_desc(fr)
        if sf:
            return _hs(9 if sf == 14 else 8, sf)
        return _hs(5, fr[0], 0, fr[1], fr[2], fr[3], fr[4] if len(fr) > 4 else 0)

    # ランクの最大カウント（早期ルーティング用）
    max_cnt = max(rc)

    # ── フォーカード ──
    if max_cnt == 4:
        for r in range(14, 1, -1):
            if rc[r] == 4:
                kicker = 0
                for k in range(14, 1, -1):
                    if k != r and rc[k] > 0:
                        kicker = k
                        break
                return _hs(7, r, 0, kicker)

    # ── フルハウス / スリーカード ──
    if max_cnt >= 3:
        t1 = t2 = 0
        for r in range(14, 1, -1):
            if rc[r] >= 3:
                if t1 == 0:
                    t1 = r
                else:
                    t2 = r
                    break

        best_pair = t2
        if best_pair == 0:
            for r in range(14, 1, -1):
                if r != t1 and rc[r] >= 2:
                    best_pair = r
                    break
        if best_pair > 0:
            return _hs(6, t1, best_pair)

    # ── ストレート（ビットマスク判定）──
    rank_mask = 0
    for r in range(2, 15):
        if rc[r] > 0:
            rank_mask |= (1 << r)

    for high in range(14, 4, -1):
        if (rank_mask >> (high - 4)) & 0b11111 == 0b11111:
            return _hs(4, high)
    # ホイール A-2-3-4-5
    wheel_mask = (1 << 14) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2)
    if (rank_mask & wheel_mask) == wheel_mask:
        return _hs(4, 5)

    # ── スリーカード（フルハウスなし）──
    if max_cnt >= 3:
        k0 = k1 = 0
        ki = 0
        for r in range(14, 1, -1):
            if rc[r] > 0 and r != t1:
                if ki == 0:
                    k0 = r
                elif ki == 1:
                    k1 = r
                    break
                ki += 1
        return _hs(3, t1, 0, k0, k1)

    # ── ツーペア / ワンペア ──
    if max_cnt == 2:
        p1 = p2 = 0
        for r in range(14, 1, -1):
            if rc[r] >= 2:
                if p1 == 0:
                    p1 = r
                elif p2 == 0:
                    p2 = r
                    break

        if p2 > 0:
            # ツーペア
            k = 0
            for r in range(14, 1, -1):
                if r != p1 and r != p2 and rc[r] > 0:
                    k = r
                    break
            return _hs(2, p1, p2, k)

        # ワンペア
        k0 = k1 = k2 = 0
        ki = 0
        for r in range(14, 1, -1):
            if r != p1 and rc[r] > 0:
                if ki == 0:
                    k0 = r
                elif ki == 1:
                    k1 = r
                elif ki == 2:
                    k2 = r
                    break
                ki += 1
        return _hs(1, p1, 0, k0, k1, k2)

    # ── ハイカード ──
    u0 = u1 = u2 = u3 = u4 = 0
    ui = 0
    for r in range(14, 1, -1):
        if rc[r] > 0:
            if ui == 0:
                u0 = r
            elif ui == 1:
                u1 = r
            elif ui == 2:
                u2 = r
            elif ui == 3:
                u3 = r
            elif ui == 4:
                u4 = r
                break
            ui += 1
    return _hs(0, u0, 0, u1, u2, u3, u4)


def calculate_equity_fast(my_hand, board, num_opponents: int,
                          num_simulations: int = 400,
                          rng: np.random.Generator | None = None) -> float:
    """
    numpy 一括サンプリング + 整数評価による高速エクイティ計算。

    my_hand: Hand オブジェクト
    board:   Board オブジェクト
    num_opponents: アクティブな相手の数
    num_simulations: モンテカルロ試行回数
    rng: numpy.random.Generator（省略時は都度生成）
    """
    if num_opponents <= 0:
        return 1.0

    # ── カードを整数に変換 ──
    my_ints: list[int] = [card_to_int(c) for c in my_hand.cards]
    board_list = board.get_all_cards()
    board_ints: list[int] = [card_to_int(c) for c in board_list]

    known_set = set(my_ints + board_ints)
    remaining = np.array([i for i in range(52) if i not in known_set], dtype=np.int32)
    n_rem = len(remaining)

    n_board_needed = 5 - len(board_ints)
    n_opp_cards = num_opponents * 2
    n_needed = n_board_needed + n_opp_cards

    if rng is None:
        rng = np.random.default_rng()

    # ── 全シミュレーション分を一括サンプリング ──
    # random float 行列を argsort して置換サンプリング（numpy ネイティブ）
    rand_vals = rng.random((num_simulations, n_rem))
    perm_idx = np.argsort(rand_vals, axis=1)[:, :n_needed]
    all_chosen = remaining[perm_idx]  # (num_simulations, n_needed) int32

    # ── 評価ループ（事前割り当てリストで list concat を回避）──
    all_chosen_list = all_chosen.tolist()  # Python list of lists（一回のみ変換）

    board_len = len(board_ints)
    # 7枚のワークスペースを1回だけ確保
    my_7 = [0, 0, 0, 0, 0, 0, 0]
    opp_7 = [0, 0, 0, 0, 0, 0, 0]
    # my_hand の2枚は不変
    my_7[0] = my_ints[0]
    my_7[1] = my_ints[1]

    wins = 0.0
    for i in range(num_simulations):
        chosen = all_chosen_list[i]

        # ボード5枚をmy_7[2..6] に直接書き込み
        ptr_b = 0
        for j in range(board_len):
            my_7[2 + ptr_b] = board_ints[j]
            ptr_b += 1
        for j in range(n_board_needed):
            my_7[2 + ptr_b] = chosen[j]
            ptr_b += 1

        my_score = evaluate_7_score(my_7)

        # 相手ハンドも同じボードを共有
        opp_7[2] = my_7[2]
        opp_7[3] = my_7[3]
        opp_7[4] = my_7[4]
        opp_7[5] = my_7[5]
        opp_7[6] = my_7[6]

        max_opp = 0
        tie_count = 0
        ptr = n_board_needed
        for _ in range(num_opponents):
            opp_7[0] = chosen[ptr]
            opp_7[1] = chosen[ptr + 1]
            opp_score = evaluate_7_score(opp_7)
            if opp_score > max_opp:
                max_opp = opp_score
                tie_count = 1
            elif opp_score == max_opp:
                tie_count += 1
            ptr += 2

        if my_score > max_opp:
            wins += 1.0
        elif my_score == max_opp:
            wins += 1.0 / (tie_count + 1)

    return wins / num_simulations
