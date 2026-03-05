"""
bayesian_strategy.py

ベイジアン手札範囲推定モジュール（numpy 実装）。

対戦相手のアクション履歴をもとに 169 手グループ（プリフロップ）または
1326 コンボ（ポストフロップ）の分布をベイズ更新し、レンジ強度を推定する。

グループインデックス:
    0-12  : ペア    (AA=0, KK=1, ..., 22=12)
    13-90 : スーテッド (AKs=13, ..., 32s=90)
    91-168: オフスーツ (AKo=91, ..., 32o=168)

GRU モデル (torch が利用可能な場合のみ) も提供するが、
ゲームプレイには numpy の BeliefTracker を使用する。
"""
from __future__ import annotations

import math
import numpy as np

from card import Card, Hand, Board, create_deck

# ──────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────

NUM_GROUPS = 169    # プリフロップ手グループ数
NUM_COMBOS = 1326   # C(52, 2) ポストフロップコンボ数
NUM_ACTIONS = 6     # fold / check / call / raise_small / raise_mid / raise_large

# ──────────────────────────────────────────────
# 169 グループ ↔ カード変換
# ──────────────────────────────────────────────

_RANK_PAIR_TO_IDX: dict[tuple[int, int], int] = {}
_idx = 0
for _r1 in range(14, 1, -1):
    for _r2 in range(_r1 - 1, 1, -1):
        _RANK_PAIR_TO_IDX[(_r1, _r2)] = _idx
        _idx += 1
# _RANK_PAIR_TO_IDX: {(14,13): 0, (14,12): 1, ..., (3,2): 77}


def hand_to_group(card1: Card, card2: Card) -> int:
    """2 枚のカードを 169 グループのインデックスに変換する。"""
    r1, r2 = card1.rank_int, card2.rank_int
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return 14 - r1          # ペア: AA=0, KK=1, ..., 22=12
    pair_idx = _RANK_PAIR_TO_IDX[(r1, r2)]
    if card1.suit == card2.suit:
        return 13 + pair_idx    # スーテッド
    return 91 + pair_idx        # オフスーツ


# ── モジュール読み込み時に全 1326 コンボを事前生成 ──────────────────
_deck = create_deck()
_ALL_COMBOS: list[tuple[Card, Card]] = [
    (_deck[i], _deck[j])
    for i in range(len(_deck))
    for j in range(i + 1, len(_deck))
]

_COMBO_TO_GROUP: np.ndarray = np.array(
    [hand_to_group(c1, c2) for c1, c2 in _ALL_COMBOS], dtype=np.int32
)  # shape: (1326,)

# カード整数インデックス（fast_eval と同じエンコード: (rank-2)*4 + suit_int）
_SUIT_TO_INT = {'s': 0, 'h': 1, 'd': 2, 'c': 3}


def _card_idx(card: Card) -> int:
    return (card.rank_int - 2) * 4 + _SUIT_TO_INT[card.suit]


_COMBO_C1_IDX = np.array([_card_idx(c1) for c1, _ in _ALL_COMBOS], dtype=np.int32)
_COMBO_C2_IDX = np.array([_card_idx(c2) for _, c2 in _ALL_COMBOS], dtype=np.int32)

# ──────────────────────────────────────────────
# グループ強度スコア (0-1)
# ──────────────────────────────────────────────


def _compute_group_strengths() -> np.ndarray:
    """各 169 グループの近似手強さを計算する（0=最弱, 1=最強）。"""
    strengths = np.zeros(NUM_GROUPS, dtype=np.float32)
    # ペア: 22=0.50, AA=1.00
    for i in range(13):
        rank = 14 - i
        strengths[i] = 0.50 + (rank - 2) / 24.0
    # スーテッド / オフスーツ
    for r1 in range(14, 1, -1):
        for r2 in range(r1 - 1, 1, -1):
            pair_idx = _RANK_PAIR_TO_IDX[(r1, r2)]
            gap = r1 - r2
            base = (r1 - 2) / 12.0 * 0.55 + (r2 - 2) / 12.0 * 0.20
            conn = max(0.0, (5 - gap) / 5.0) * 0.08
            strengths[13 + pair_idx] = float(min(base + conn + 0.05, 0.95))  # suited
            strengths[91 + pair_idx] = float(min(base + conn,       0.90))   # offsuit
    return strengths


_GROUP_STRENGTHS: np.ndarray = _compute_group_strengths()

# ──────────────────────────────────────────────
# PlayerProfile（ラウンドを跨いだプレイヤー行動統計）
# ──────────────────────────────────────────────


class PlayerProfile:
    """
    複数ラウンドを跨いで対戦相手の行動傾向を学習するプロファイル。

    統計量:
        VPIP : プリフロップ自発参加率（call/raise した割合）
        PFR  : プリフロップレイズ率
        AF   : ポストフロップ攻撃性 = raises / (calls + checks + folds)
        WTSD : ショーダウン到達率

    これらを使って尤度関数を個人化し、fold 情報を適切に活用する。
    """

    MIN_HANDS = 5  # プロファイルを信頼するための最低観測ハンド数

    def __init__(self) -> None:
        self.hands_dealt: int = 0
        self._vpip: int = 0         # preflop call / raise 回数
        self._pfr: int = 0          # preflop raise 回数
        self._pf_raise: int = 0     # postflop raise カウント
        self._pf_passive: int = 0   # postflop call / check カウント
        self._pf_fold: int = 0      # postflop fold カウント
        self._sd_count: int = 0     # ショーダウン到達回数
        self._sd_strength_sum: float = 0.0  # ショーダウン時の手強度の合計

    # ── 記録 ──────────────────────────────────────────────────────────────

    def record_preflop_action(self, action: str) -> None:
        """ラウンド最初のプリフロップアクションを記録する。"""
        self.hands_dealt += 1
        if action in ('call', 'raise'):
            self._vpip += 1
        if action == 'raise':
            self._pfr += 1

    def record_postflop_action(self, action: str) -> None:
        """ポストフロップのアクションを記録する。"""
        if action == 'raise':
            self._pf_raise += 1
        elif action == 'fold':
            self._pf_fold += 1
        else:
            self._pf_passive += 1

    def record_showdown_hand(self, group_idx: int) -> None:
        """ショーダウンで見えた手グループのインデックスを記録する。"""
        self._sd_count += 1
        self._sd_strength_sum += float(_GROUP_STRENGTHS[group_idx])

    # ── 統計プロパティ ─────────────────────────────────────────────────────

    @property
    def has_profile(self) -> bool:
        return self.hands_dealt >= self.MIN_HANDS

    @property
    def vpip_rate(self) -> float:
        """プリフロップ自発参加率（0-1）。データ不足時は 0.5。"""
        return self._vpip / max(self.hands_dealt, 1)

    @property
    def pfr_rate(self) -> float:
        return self._pfr / max(self.hands_dealt, 1)

    @property
    def aggression_factor(self) -> float:
        """AF = raises / (calls + checks + folds)。"""
        denom = self._pf_passive + self._pf_fold
        return self._pf_raise / max(denom, 1)

    @property
    def postflop_fold_rate(self) -> float:
        total = self._pf_raise + self._pf_passive + self._pf_fold
        return self._pf_fold / max(total, 1)

    @property
    def avg_showdown_strength(self) -> float:
        """ショーダウン時の平均手強度。データなしは 0.6（プレイヤーの典型値）。"""
        if self._sd_count == 0:
            return 0.6
        return self._sd_strength_sum / self._sd_count

    def __repr__(self) -> str:
        return (
            f"PlayerProfile(hands={self.hands_dealt}, VPIP={self.vpip_rate:.0%}, "
            f"PFR={self.pfr_rate:.0%}, AF={self.aggression_factor:.1f})"
        )


# ──────────────────────────────────────────────
# デッドカードマスク（ベクトル化）
# ──────────────────────────────────────────────


def build_dead_mask(dead_cards: list[Card]) -> np.ndarray:
    """デッドカードを含むコンボのブールマスク shape: (1326,)。True=使えない。"""
    dead_arr = np.zeros(52, dtype=bool)
    for c in dead_cards:
        dead_arr[_card_idx(c)] = True
    return dead_arr[_COMBO_C1_IDX] | dead_arr[_COMBO_C2_IDX]

# ──────────────────────────────────────────────
# ベイズ更新用の尤度関数
# ──────────────────────────────────────────────


def _log_likelihood_groups(
    action: str,
    amount: int,
    pot: int,
    profile: "PlayerProfile | None" = None,
) -> np.ndarray:
    """
    P(action | group) の対数尤度 shape: (169,)。プロファイルで個人化する。

    amount: 実際に賭けた額 (fold=0, check=0, call=call_amount, raise=raise_to_amount)
    pot:    アクション前のポット
    profile: 対戦相手のプロファイル（None の場合はデフォルト値を使用）

    プロファイル活用の考え方:
        VPIP が低い（タイト）プレイヤー:
          - fold → 非常に弱い手の強い証拠（強い手はプレイするはず）
          - call → 強い手の強い証拠（通常は弱い手でプレイしない）
        VPIP が高い（ルーズ）プレイヤー:
          - fold → まずまずの手でもありうる（判断基準が甘い）
          - call → 弱い手でもコールしうる
        AF が高い（アグレッシブ）プレイヤー:
          - raise → ブラフ比率が高い（二極化が強い）
    """
    s = _GROUP_STRENGTHS.astype(np.float64)   # (169,)
    bet_ratio = float(amount) / max(float(pot), 1.0)

    # プロファイルから統計量を取得（データ不足時はデフォルト）
    if profile is not None and profile.has_profile:
        vpip = profile.vpip_rate
        af = profile.aggression_factor
    else:
        vpip = 0.5   # 中間的なプレイヤーと仮定
        af = 1.0

    if action == 'fold':
        # タイトなプレイヤー（低VPIP）の fold は「非常に弱い手」の強い証拠
        # fold_sharpness: VPIP=0.2→3.4, VPIP=0.5→1.0, VPIP=0.8→0.4
        fold_sharpness = max(0.4, (1.0 - vpip) * 4.0 - 0.6)
        raw = (1.0 - s) ** fold_sharpness + 0.02

    elif action == 'check':
        # タイトなプレイヤーのチェックは「中程度」の手（強い手はベットする）
        raw = 0.10 + 0.80 * np.clip(1.0 - (s - 0.45) ** 2 * 4.0, 0, 1)

    elif action == 'call':
        # タイトなプレイヤーのコールはより強い手の証拠
        # strength_exponent: VPIP=0.2→2.4, VPIP=0.5→1.5, VPIP=0.8→0.6
        strength_exponent = max(0.5, (1.0 - vpip) * 3.0)
        factor = max(0.15, 1.0 - bet_ratio * 0.5)
        raw = factor * (0.05 + s ** strength_exponent)

    else:  # raise / bet
        # AF が高いプレイヤーはブラフ的なレイズが多い（二極化を強める）
        # bluff_share: AF=0.5→0.2, AF=1→0.25, AF=3→0.38
        bluff_share = min(af / (af + 3.0), 0.4)
        value_share = 1.0 - bluff_share
        if bet_ratio > 1.0:
            # 大きなベット: 価値レイズ + ブラフの二極化
            raw = 0.03 + value_share * 0.65 * s ** 2 + bluff_share * 0.25 * (1.0 - s) ** 2
        else:
            # 通常サイズ: 主に価値レイズ
            raw = 0.03 + value_share * 0.70 * s + bluff_share * 0.15

    return np.log(np.clip(raw, 1e-6, None))


def _log_likelihood_combos(
    action: str,
    amount: int,
    pot: int,
    profile: "PlayerProfile | None" = None,
) -> np.ndarray:
    """P(action | combo) の対数尤度 shape: (1326,)。グループ尤度をコンボに展開する。"""
    return _log_likelihood_groups(action, amount, pot, profile)[_COMBO_TO_GROUP]

# ──────────────────────────────────────────────
# 正規化ユーティリティ
# ──────────────────────────────────────────────


def _log_normalize(log_p: np.ndarray) -> np.ndarray:
    """-inf を除いた有効エントリで対数確率を正規化する。"""
    out = log_p.copy()
    valid = ~np.isneginf(out)
    if not np.any(valid):
        return out
    lp = out[valid]
    lp -= np.max(lp)
    lp -= np.log(np.sum(np.exp(lp)))
    out[valid] = lp
    return out


def _log_normalize_full(log_p: np.ndarray) -> np.ndarray:
    """全要素が有効な 1D ベクトルを正規化する（-inf なし前提）。"""
    lp = log_p - np.max(log_p)
    lp -= np.log(np.sum(np.exp(lp)))
    return lp

# ──────────────────────────────────────────────
# BeliefTracker
# ──────────────────────────────────────────────


class BeliefTracker:
    """
    対戦相手 1 人のハンド分布を追跡するベイジアントラッカー。

    - プリフロップ: 169 グループの均等事前分布で開始
    - フロップ以降: 1326 コンボに展開しデッドカードでマスク
    """

    def __init__(self) -> None:
        self._log_p: np.ndarray = np.full(
            NUM_GROUPS, -math.log(NUM_GROUPS), dtype=np.float64
        )
        self._is_postflop: bool = False
        self._log_p_1326: np.ndarray | None = None
        self._dead_mask: np.ndarray | None = None  # shape: (1326,) bool

    @property
    def is_postflop(self) -> bool:
        return self._is_postflop

    # ── ストリート移行 ──────────────────────────────────────────────────

    def expand_to_postflop(self, dead_cards: list[Card]) -> None:
        """フロップ時に 169 グループ → 1326 コンボ分布に展開する。"""
        if self._is_postflop:
            return

        # グループ確率をコンボに均等割り
        group_sizes = np.bincount(_COMBO_TO_GROUP, minlength=NUM_GROUPS).astype(np.float64)
        log_p_1326 = (
            self._log_p[_COMBO_TO_GROUP]
            - np.log(group_sizes[_COMBO_TO_GROUP] + 1e-12)
        )

        # デッドカードマスク
        mask = build_dead_mask(dead_cards)
        log_p_1326[mask] = -np.inf

        self._log_p_1326 = _log_normalize(log_p_1326)
        self._dead_mask = mask.copy()
        self._is_postflop = True

    def apply_new_dead_cards(self, dead_cards: list[Card]) -> None:
        """ターン / リバーで新たに公開されたカードを反映する。"""
        if not self._is_postflop or self._log_p_1326 is None or self._dead_mask is None:
            return
        new_mask = build_dead_mask(dead_cards)
        added = new_mask & ~self._dead_mask
        if np.any(added):
            self._log_p_1326[added] = -np.inf
            self._log_p_1326 = _log_normalize(self._log_p_1326)
            self._dead_mask |= added

    # ── ベイズ更新 ──────────────────────────────────────────────────────

    def update(
        self,
        action: str,
        amount: int,
        pot: int,
        dead_cards: list[Card] | None = None,
        profile: "PlayerProfile | None" = None,
    ) -> None:
        """
        観測したアクションでベイズ更新する。

        amount:    実際に賭けた額 (fold/check=0, call=call_amount, raise=raise_to_amount)
        pot:       アクション前のポット
        dead_cards: ボード + 自分の手牌（ポストフロップ時に渡す）
        profile:   対戦相手プロファイル（尤度の個人化に使用）
        """
        if not self._is_postflop:
            ll = _log_likelihood_groups(action, amount, pot, profile)
            self._log_p = _log_normalize_full(self._log_p + ll)
        else:
            if dead_cards is not None:
                self.apply_new_dead_cards(dead_cards)
            if self._log_p_1326 is None:
                return
            ll = _log_likelihood_combos(action, amount, pot, profile)
            self._log_p_1326 = _log_normalize(self._log_p_1326 + ll)

    # ── レンジ統計 ──────────────────────────────────────────────────────

    def mean_strength(self) -> float:
        """信念分布の加重平均手強さ（0-1）を返す。"""
        if not self._is_postflop:
            lp = self._log_p - np.max(self._log_p)
            probs = np.exp(lp)
            probs /= probs.sum()
            return float(np.dot(probs, _GROUP_STRENGTHS))

        if self._log_p_1326 is None:
            return 0.5
        probs = np.where(np.isneginf(self._log_p_1326), 0.0, np.exp(self._log_p_1326))
        total = probs.sum()
        if total <= 0:
            return 0.5
        probs /= total
        return float(np.dot(probs, _GROUP_STRENGTHS[_COMBO_TO_GROUP]))


# ──────────────────────────────────────────────
# GRU 対戦相手モデル（torch が利用可能な場合のみ）
# ──────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    class GRUOpponentActionModel(nn.Module):  # type: ignore[no-redef]
        """
        GRU ベースの対戦相手アクションモデル。

        入力:
            state_vector:         [1, state_dim]         - ゲーム状態特徴量
            hand_features_matrix: [num_hands, hand_dim]  - ハンド特徴量
            hidden:               [1, num_hands, hidden_dim]

        出力:
            log_action_probs: [num_hands, NUM_ACTIONS]
            hidden:           [1, num_hands, hidden_dim]
        """

        def __init__(
            self,
            state_dim: int = 10,
            hand_feature_dim: int = 5,
            hidden_dim: int = 128,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.gru = nn.GRU(state_dim + hand_feature_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, NUM_ACTIONS)

        def forward(self, state_vector, hand_features_matrix, hidden):
            num_hands = hand_features_matrix.size(0)
            expanded_state = state_vector.expand(num_hands, -1)
            x = torch.cat([expanded_state, hand_features_matrix], dim=1).unsqueeze(1)
            out, hidden = self.gru(x, hidden)
            logits = self.fc(out[:, 0, :])
            return F.log_softmax(logits, dim=1), hidden

        def init_hidden(self, num_hands: int):
            return torch.zeros(1, num_hands, self.hidden_dim)

    def build_hand_features(
        combos: list[tuple[Card, Card]] | None = None,
    ) -> "torch.Tensor":
        """
        ハンドコンボの特徴量行列を構築する shape: (num_combos, 5)。

        特徴量: [高いランク/14, 低いランク/14, スーテッド, ペア, コネクタ度]
        """
        if combos is None:
            combos = _ALL_COMBOS
        feats = []
        for c1, c2 in combos:
            r1, r2 = c1.rank_int, c2.rank_int
            if r1 < r2:
                r1, r2 = r2, r1
            is_pair = float(r1 == r2)
            is_suited = float(c1.suit == c2.suit and r1 != r2)
            gap = (r1 - r2) if r1 != r2 else 0
            connectivity = max(0.0, (5 - gap) / 5.0)
            feats.append([r1 / 14.0, r2 / 14.0, is_suited, is_pair, connectivity])
        return torch.tensor(feats, dtype=torch.float32)

    def build_state_vector(
        street: int,
        pot: int,
        call_amount: int,
        num_active: int,
        total_players: int,
        stack_ratio: float = 1.0,
    ) -> "torch.Tensor":
        """
        ゲーム状態を 10 次元特徴量ベクトルに変換する shape: (1, 10)。

        特徴量: [street/3, pot/1000, pot_odds, active/total, stack_ratio, 0*5]
        """
        pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 else 0.0
        return torch.tensor([[
            street / 3.0,
            min(pot / 1000.0, 1.0),
            pot_odds,
            num_active / max(total_players, 1),
            min(stack_ratio, 1.0),
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]], dtype=torch.float32)

    def process_hand_episode(
        action_history: list[dict],
        is_showdown: bool,
        model: "GRUOpponentActionModel",
        optimizer,
        final_true_hand_idx_1326: int | None = None,
    ) -> float:
        """
        1 ハンドの行動列からモデルをオフライン学習する。

        action_history の各ステップ:
            street:          int  (0=pre, 1=flop, 2=turn, 3=river)
            state:           torch.Tensor [1, state_dim]
            features:        torch.Tensor [num_hands, hand_feature_dim]
            action:          int  (0=fold,1=check/call,2-5=raise)
            dead_cards_mask: torch.Tensor [1326] bool  (ポストフロップのみ)

        is_showdown: ショーダウンか否か
        final_true_hand_idx_1326: ショーダウン時の実際の手のコンボインデックス
        """
        _combo_to_group_t = torch.from_numpy(_COMBO_TO_GROUP)
        _group_sizes_t = torch.bincount(_combo_to_group_t, minlength=NUM_GROUPS).float()

        criterion_showdown = nn.NLLLoss()
        criterion_fold = nn.KLDivLoss(reduction='batchmean')

        optimizer.zero_grad()

        current_num_hands = NUM_GROUPS
        log_belief = torch.full((NUM_GROUPS,), -math.log(NUM_GROUPS))
        hidden = model.init_hidden(NUM_GROUPS)

        for step_data in action_history:
            street = step_data['street']

            # フロップ移行: 169 グループ → 1326 コンボに展開
            if street >= 1 and current_num_hands == NUM_GROUPS:
                dead_mask = step_data['dead_cards_mask']  # [1326] bool tensor
                log_p_1326 = (
                    log_belief[_combo_to_group_t]
                    - torch.log(_group_sizes_t[_combo_to_group_t] + 1e-9)
                )
                log_p_1326 = log_p_1326.masked_fill(dead_mask, float('-inf'))
                log_belief = F.log_softmax(log_p_1326, dim=0)
                # 隠れ状態も 169 → 1326 に展開
                hidden = hidden[:, _combo_to_group_t, :]
                current_num_hands = NUM_COMBOS

            state_vector = step_data['state']    # [1, state_dim]
            hand_features = step_data['features']  # [num_hands, hand_feature_dim]
            actual_action = step_data['action']    # 0-5

            log_action_probs, hidden = model(state_vector, hand_features, hidden)
            log_belief = log_belief + log_action_probs[:, actual_action]
            log_belief = F.log_softmax(log_belief, dim=0)

        # 損失計算と学習
        if is_showdown and final_true_hand_idx_1326 is not None:
            target = torch.tensor([final_true_hand_idx_1326], dtype=torch.long)
            loss = criterion_showdown(log_belief.unsqueeze(0), target)
        else:
            target_dist = torch.exp(log_belief).detach()
            loss = criterion_fold(log_belief.unsqueeze(0), target_dist.unsqueeze(0))

        loss.backward()
        optimizer.step()
        return loss.item()