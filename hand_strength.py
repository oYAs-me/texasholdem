from itertools import combinations
from collections import Counter
from card import Card, Hand, Board

# ポーカーハンドのランクを定義（強い順）
# 各タプルは (ハンドタイプ, primary_rank, secondary_rank, kicker_ranks...)
# ハンドタイプは整数で、0-9 の範囲で、0がハイカード、9がロイヤルフラッシュ
HAND_RANK_MAP = {
  'HIGH_CARD': 0,           # ハイカード
  'ONE_PAIR': 1,            # ワンペア
  'TWO_PAIR': 2,            # ツーペア
  'THREE_OF_A_KIND': 3,     # スリーカード
  'STRAIGHT': 4,            # ストレート
  'FLUSH': 5,               # フラッシュ
  'FULL_HOUSE': 6,          # フルハウス
  'FOUR_OF_A_KIND': 7,      # フォーカード
  'STRAIGHT_FLUSH': 8,      # ストレートフラッシュ
  'ROYAL_FLUSH': 9          # ロイヤルフラッシュ
}

class EvaluatedHand:
  """評価されたポーカーハンドを表すクラス"""
  def __init__(self, hand_type: str, primary_rank: int = 0, secondary_rank: int = 0, kicker_ranks: tuple = ()):
    self.hand_type = hand_type
    self.hand_type_rank = HAND_RANK_MAP[hand_type]
    self.primary_rank = primary_rank           # メインのカードランク
    self.secondary_rank = secondary_rank       # サブのカードランク（ツーペアなど用）
    self.kicker_ranks = kicker_ranks           # キッカー（追加のランク）
    self.value = self._calculate_value()       # 比較用の数値を計算

  def _calculate_value(self):
    """ハンドの強さを数値化する"""
    value_components = []
    value_components.append(self.hand_type_rank)
    value_components.append(self.primary_rank)
    value_components.append(self.secondary_rank)

    # キッカーを4スロット分に正規化（不足分は0で埋める）
    expected_kicker_slots = 4
    current_kicker_count = len(self.kicker_ranks)

    for i in range(expected_kicker_slots):
      if i < current_kicker_count:
        value_components.append(self.kicker_ranks[i])
      else:
        value_components.append(0)

    # 各コンポーネントを進数表記で結合して単一の比較可能な値にする
    final_value = 0
    # 右から左へ反復（最下位から最上位）
    for i, comp in enumerate(reversed(value_components)):
      # 各コンポーネントの最大値は14なので、15進法で分離
      final_value += comp * (15 ** i)

    return final_value

  def __lt__(self, other):
    """より弱いハンドかどうかを判定"""
    return self.value < other.value

  def __eq__(self, other):
    """同じ強さのハンドかどうかを判定"""
    return self.value == other.value

  def __gt__(self, other):
    """より強いハンドかどうかを判定"""
    return self.value > other.value

  def __str__(self):
    """ハンド情報を文字列で返す"""
    kickers_str = ', '.join(str(r) for r in self.kicker_ranks)
    return f"{self.hand_type} (Primary: {self.primary_rank}, Secondary: {self.secondary_rank}, Kickers: [{kickers_str}])"

def evaluate_hand(hand: Hand, board: Board) -> EvaluatedHand:
  """ハンドとボードから役を評価して、EvaluatedHandオブジェクトを返す"""

  BOARD = (list(board.flops) if board.flops else []) + ([board.turn] if board.turn else []) + ([board.river] if board.river else []) # community cardsは0-5枚の可能性がある
  ALL_CARDS = list(hand.cards) + BOARD # all_cardsは2-7枚の可能性がある

  INT_COUNTER = Counter(card.rank_int for card in ALL_CARDS)
  SUIT_COUNTER = Counter(card.suit for card in ALL_CARDS)

  # 以下の役判定は2-7枚のカードで実行されても正しく機能するように設計

  def is_pair() -> int | None:
    """ワンペアの判定"""
    pairs = [rank for rank, count in INT_COUNTER.items() if count == 2]
    if pairs:
      return max(pairs)  # 最も強いペアのランクを返す
    return None
  
  def is_two_pair() -> tuple[int, int] | None:
    """ツーペアの判定"""
    pairs = [rank for rank, count in INT_COUNTER.items() if count == 2]
    if len(pairs) >= 2:
      top_two = sorted(pairs, reverse=True)[:2]
      return top_two[0], top_two[1]  # 強いペアと弱いペアのランクを返す
    return None
  
  def is_three_of_a_kind() -> int | None:
    """スリーカードの判定"""
    threes = [rank for rank, count in INT_COUNTER.items() if count == 3]
    if threes:
      return max(threes)  # 最も強いスリーのランクを返す
    return None
  
  def is_straight() -> int | None:
    """ストレートの判定"""
    if len(INT_COUNTER) < 5:
      return None  # ユニークなランクが5未満ならストレートは成立しない
    unique_ranks = sorted(set(INT_COUNTER.keys()), reverse=True)
    # A-5のストレートも考慮
    if 14 in unique_ranks:
      unique_ranks.append(1)  # Aを1としても扱う

    for i in range(len(unique_ranks) - 4):
      if (unique_ranks[i] - unique_ranks[i + 4]) == 4:
        return unique_ranks[i]  # ストレートの最高ランクを返す
    return None
  
  def is_flush() -> list[int] | None:
    """フラッシュの判定\n
    フラッシュが成立している場合は、フラッシュのカードランクを降順のリストで返す"""
    for suit, count in SUIT_COUNTER.items():
      if count >= 5:
        flush_cards = [card.rank_int for card in ALL_CARDS if card.suit == suit]
        top_five = sorted(flush_cards, reverse=True)[:5]
        return top_five  # フラッシュのカードランクを返す
    return None
  
  def is_full_house() -> tuple[int, int] | None:
    """フルハウスの判定\n
    フルハウスが成立している場合は、スリーカードのランクとペアのランクをこの順で返す"""
    three = is_three_of_a_kind()
    if three:
      # スリーカードのランクを除外してペアを探す
      remaining_ranks = {rank: count for rank, count in INT_COUNTER.items() if rank != three}
      pairs = [rank for rank, count in remaining_ranks.items() if count >= 2]
      if pairs:
        return three, max(pairs)  # スリーのランクと最も強いペアのランクを返す
    return None
  
  def is_four_of_a_kind() -> int | None:
    """フォーカードの判定"""
    fours = [rank for rank, count in INT_COUNTER.items() if count == 4]
    if fours:
      return max(fours)  # 最も強いフォーのランクを返す
    return None
  
  def is_straight_flush() -> int | None:
    """ストレートフラッシュの判定"""
    flush_ranks = is_flush()
    if flush_ranks:
      # フラッシュのカードだけでストレートが成立するか確認
      unique_flush_ranks = sorted(set(flush_ranks), reverse=True)
      # A-5のストレートも考慮
      if 14 in unique_flush_ranks:
        unique_flush_ranks.append(1)  # Aを1としても扱う

      for i in range(len(unique_flush_ranks) - 4):
        if (unique_flush_ranks[i] - unique_flush_ranks[i + 4]) == 4:
          return unique_flush_ranks[i]  # ストレートフラッシュの最高ランクを返す
    return None
  
  def is_royal_flush() -> int | None:
    """ロイヤルフラッシュの判定"""
    straight_flush_rank = is_straight_flush()
    if straight_flush_rank == 14:  # ストレートフラッシュの最高ランクがAならロイヤルフラッシュ
      return straight_flush_rank
    return None
  
  # ハンドの強さを判定する順番は、ロイヤルフラッシュ → ストレートフラッシュ → フォーカード → フルハウス → フラッシュ → ストレート → スリーカード → ツーペア → ワンペア → ハイカード
  if (royal_flush_rank := is_royal_flush()) is not None:
    return EvaluatedHand('ROYAL_FLUSH', primary_rank=royal_flush_rank)
  if (straight_flush_rank := is_straight_flush()) is not None:
    return EvaluatedHand('STRAIGHT_FLUSH', primary_rank=straight_flush_rank)
  if (four_of_a_kind_rank := is_four_of_a_kind()) is not None:
    kicker_ranks = sorted((rank for rank in INT_COUNTER.keys() if rank != four_of_a_kind_rank), reverse=True)[:1]  # フォーカード以外のカードの中で最も強いものをキッカーとして1枚だけ選ぶ
    return EvaluatedHand('FOUR_OF_A_KIND', primary_rank=four_of_a_kind_rank, kicker_ranks=tuple(kicker_ranks))
  if (full_house_ranks := is_full_house()) is not None:
    return EvaluatedHand('FULL_HOUSE', primary_rank=full_house_ranks[0], secondary_rank=full_house_ranks[1])
  if (flush_ranks := is_flush()) is not None:
    return EvaluatedHand('FLUSH', primary_rank=flush_ranks[0], kicker_ranks=tuple(flush_ranks[1:]))
  if (straight_rank := is_straight()) is not None:
    return EvaluatedHand('STRAIGHT', primary_rank=straight_rank)
  if (three_of_a_kind_rank := is_three_of_a_kind()) is not None:
    kicker_ranks = sorted((rank for rank in INT_COUNTER.keys() if rank != three_of_a_kind_rank), reverse=True)[:2]  # スリーカード以外のカードの中で最も強いものをキッカーとして2枚だけ選ぶ
    return EvaluatedHand('THREE_OF_A_KIND', primary_rank=three_of_a_kind_rank, kicker_ranks=tuple(kicker_ranks))
  if (two_pair_ranks := is_two_pair()) is not None:
    kicker_ranks = sorted((rank for rank in INT_COUNTER.keys() if rank != two_pair_ranks[0] and rank != two_pair_ranks[1]), reverse=True)[:1]  # ツーペア以外のカードの中で最も強いものをキッカーとして1枚だけ選ぶ
    return EvaluatedHand('TWO_PAIR', primary_rank=two_pair_ranks[0], secondary_rank=two_pair_ranks[1], kicker_ranks=tuple(kicker_ranks))
  if (one_pair_rank := is_pair()) is not None:
    kicker_ranks = sorted((rank for rank in INT_COUNTER.keys() if rank != one_pair_rank), reverse=True)[:3]  # ワンペア以外のカードの中で最も強いものをキッカーとして3枚だけ選ぶ
    return EvaluatedHand('ONE_PAIR', primary_rank=one_pair_rank, kicker_ranks=tuple(kicker_ranks))
  # ハイカードの場合は、キッカーとして全てのカードのランクを降順で選ぶ
  kicker_ranks = sorted(INT_COUNTER.keys(), reverse=True)[:5]  # ハイカード以外のカードの中で最も強いものをキッカーとして5枚だけ選ぶ
  return EvaluatedHand('HIGH_CARD', primary_rank=kicker_ranks[0], kicker_ranks=tuple(kicker_ranks[1:]))
  