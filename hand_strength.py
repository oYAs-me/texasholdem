from collections import Counter
from card import Card, Hand, Board

HAND_RANK_MAP = {
  'HIGH_CARD': 0, 'ONE_PAIR': 1, 'TWO_PAIR': 2, 'THREE_OF_A_KIND': 3,
  'STRAIGHT': 4, 'FLUSH': 5, 'FULL_HOUSE': 6, 'FOUR_OF_A_KIND': 7,
  'STRAIGHT_FLUSH': 8, 'ROYAL_FLUSH': 9
}

class EvaluatedHand:
  def __init__(self, hand_type: str, primary_rank: int = 0, secondary_rank: int = 0, 
               kicker_ranks: tuple = (), best_cards: list[Card] = None):
    self.hand_type = hand_type
    self.hand_type_rank = HAND_RANK_MAP[hand_type]
    self.primary_rank = primary_rank
    self.secondary_rank = secondary_rank
    self.kicker_ranks = kicker_ranks
    self.best_cards = best_cards if best_cards else []
    self.value = self._calculate_value()

  def _calculate_value(self):
    value_components = [self.hand_type_rank, self.primary_rank, self.secondary_rank]
    expected_kicker_slots = 4
    for i in range(expected_kicker_slots):
      value_components.append(self.kicker_ranks[i] if i < len(self.kicker_ranks) else 0)
    final_value = 0
    for i, comp in enumerate(reversed(value_components)):
      final_value += comp * (15 ** i)
    return final_value

  def __lt__(self, other): return self.value < other.value
  def __eq__(self, other): return self.value == other.value
  def __gt__(self, other): return self.value > other.value

def evaluate_hand(hand: Hand, board: Board) -> EvaluatedHand:
  """ハンドとボードから役を評価する（高速なルールベース判定）"""
  BOARD = (list(board.flops) if board.flops else []) + ([board.turn] if board.turn else []) + ([board.river] if board.river else [])
  ALL_CARDS = list(hand.cards) + BOARD
  
  # ランクとスートのカウント
  INT_COUNTER = Counter(card.rank_int for card in ALL_CARDS)
  SUIT_COUNTER = Counter(card.suit for card in ALL_CARDS)
  
  # 1. フラッシュ判定
  flush_suit = None
  for suit, count in SUIT_COUNTER.items():
      if count >= 5:
          flush_suit = suit
          break
  
  if flush_suit:
      flush_cards = sorted([c for c in ALL_CARDS if c.suit == flush_suit], key=lambda x: x.rank_int, reverse=True)
      # ストレートフラッシュ判定
      unique_flush_ranks = sorted(list(set(c.rank_int for c in flush_cards)), reverse=True)
      sf_high = None
      if len(unique_flush_ranks) >= 5:
          for i in range(len(unique_flush_ranks) - 4):
              if unique_flush_ranks[i] - unique_flush_ranks[i+4] == 4:
                  sf_high = unique_flush_ranks[i]
                  break
          if sf_high is None and set([14, 5, 4, 3, 2]).issubset(set(unique_flush_ranks)):
              sf_high = 5
      
      if sf_high:
          best = [c for c in flush_cards if c.rank_int <= sf_high][:5]
          if sf_high == 5: # A-5
              best = [c for c in flush_cards if c.rank_int in [14, 5, 4, 3, 2]]
          if sf_high == 14: return EvaluatedHand('ROYAL_FLUSH', 14, best_cards=best)
          return EvaluatedHand('STRAIGHT_FLUSH', sf_high, best_cards=best)
      
      # 通常のフラッシュ
      return EvaluatedHand('FLUSH', flush_cards[0].rank_int, kicker_ranks=tuple(c.rank_int for c in flush_cards[1:5]), best_cards=flush_cards[:5])

  # 2. フォーカード
  fours = [r for r, c in INT_COUNTER.items() if c == 4]
  if fours:
      f_rank = max(fours)
      kicker = max([r for r in INT_COUNTER.keys() if r != f_rank])
      best = [c for c in ALL_CARDS if c.rank_int == f_rank] + [next(c for c in ALL_CARDS if c.rank_int == kicker)]
      return EvaluatedHand('FOUR_OF_A_KIND', f_rank, kicker_ranks=(kicker,), best_cards=best)

  # 3. フルハウス
  threes = sorted([r for r, c in INT_COUNTER.items() if c == 3], reverse=True)
  pairs = sorted([r for r, c in INT_COUNTER.items() if c >= 2], reverse=True)
  if len(threes) >= 2:
      best = [c for c in ALL_CARDS if c.rank_int == threes[0]][:3] + [c for c in ALL_CARDS if c.rank_int == threes[1]][:2]
      return EvaluatedHand('FULL_HOUSE', threes[0], threes[1], best_cards=best)
  if len(threes) == 1:
      p_candidates = [p for p in pairs if p != threes[0]]
      if p_candidates:
          best = [c for c in ALL_CARDS if c.rank_int == threes[0]] + [c for c in ALL_CARDS if c.rank_int == p_candidates[0]][:2]
          return EvaluatedHand('FULL_HOUSE', threes[0], p_candidates[0], best_cards=best)

  # 4. ストレート
  unique_ranks = sorted(list(INT_COUNTER.keys()), reverse=True)
  straight_high = None
  if len(unique_ranks) >= 5:
      for i in range(len(unique_ranks) - 4):
          if unique_ranks[i] - unique_ranks[i+4] == 4:
              straight_high = unique_ranks[i]
              break
      if straight_high is None and set([14, 5, 4, 3, 2]).issubset(set(unique_ranks)):
          straight_high = 5
  
  if straight_high:
      if straight_high == 5:
          target_ranks = [14, 5, 4, 3, 2]
      else:
          target_ranks = list(range(straight_high, straight_high - 5, -1))
      best = [next(c for c in ALL_CARDS if c.rank_int == r) for r in target_ranks]
      return EvaluatedHand('STRAIGHT', straight_high, best_cards=best)

  # 5. スリーカード
  if threes:
      t_rank = threes[0]
      kickers = sorted([r for r in INT_COUNTER.keys() if r != t_rank], reverse=True)[:2]
      best = [c for c in ALL_CARDS if c.rank_int == t_rank] + [next(c for c in ALL_CARDS if c.rank_int == k) for k in kickers]
      return EvaluatedHand('THREE_OF_A_KIND', t_rank, kicker_ranks=tuple(kickers), best_cards=best)

  # 6. ツーペア
  if len(pairs) >= 2:
      p1, p2 = pairs[0], pairs[1]
      kicker = max([r for r in INT_COUNTER.keys() if r != p1 and r != p2])
      best = [c for c in ALL_CARDS if c.rank_int == p1][:2] + [c for c in ALL_CARDS if c.rank_int == p2][:2] + [next(c for c in ALL_CARDS if c.rank_int == kicker)]
      return EvaluatedHand('TWO_PAIR', p1, p2, kicker_ranks=(kicker,), best_cards=best)

  # 7. ワンペア
  if pairs:
      p_rank = pairs[0]
      kickers = sorted([r for r in INT_COUNTER.keys() if r != p_rank], reverse=True)[:3]
      best = [c for c in ALL_CARDS if c.rank_int == p_rank] + [next(c for c in ALL_CARDS if c.rank_int == k) for k in kickers]
      return EvaluatedHand('ONE_PAIR', p_rank, kicker_ranks=tuple(kickers), best_cards=best)

  # 8. ハイカード
  sorted_ranks = sorted(list(INT_COUNTER.keys()), reverse=True)[:5]
  best = [next(c for c in ALL_CARDS if c.rank_int == r) for r in sorted_ranks]
  return EvaluatedHand('HIGH_CARD', sorted_ranks[0], kicker_ranks=tuple(sorted_ranks[1:]), best_cards=best)
