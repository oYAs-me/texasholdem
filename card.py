# テキサスホールデムでの役を判定するためのクラスを作成する

SUITS_MAP = {'s': 'スペード', 'h': 'ハート', 'd': 'ダイヤ', 'c': 'クラブ'}
RANKS_MAP = {
    14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
    9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'
}
RANKS_STR_TO_INT = {v: k for k, v in RANKS_MAP.items()}

class Card:
  """1枚のトランプカードを表すクラス"""
  def __init__(self, suit: str, rank_int: int):
    """
    suit: スート ('s', 'h', 'd', 'c')
    rank_int: ランクの数値 (2-14, A=14, K=13, Q=12, J=11, T=10)
    """
    if suit not in SUITS_MAP:
      raise ValueError(f"無効なスートです: {suit}")
    if rank_int not in RANKS_MAP:
      raise ValueError(f"無効なランクです: {rank_int}")
    
    self._suit = suit
    self._rank_int = rank_int

  @property
  def suit(self) -> str:
    """スペード(s)、ハート(h)、ダイヤ(d)、クラブ(c)のいずれかを返す"""
    return self._suit

  @property
  def rank_str(self) -> str:
    """A,2,3,4,5,6,7,8,9,T,J,Q,Kのいずれかを返す"""
    return RANKS_MAP[self._rank_int]

  @property
  def rank_int(self) -> int:
    """Aは14、Kは13、Qは12、Jは11、Tは10、2-9はそのままの数値を返す"""
    return self._rank_int

  def __str__(self) -> str:
    """Cardクラスのインスタンスを文字列に変換するためのメソッド\n
    例: `str(Card('s', 2))` は `s2` になる"""
    return self.suit + self.rank_str

  def __repr__(self) -> str:
    """Cardクラスのインスタンスを文字列で表現するためのメソッド\n
    例: `repr(Card('s', 2))` は `Card('s', 2)` になる"""
    return f"Card('{self.suit}', {self.rank_int})"
  
  def __lt__(self, other) -> bool:
    """Cardクラスのインスタンス同士を比較するためのメソッド\n
    スートに関係なく、ランクの大小を比較する\n
    例: `Card('s', 14) < Card('s', 2)` は `False` になる"""
    if not isinstance(other, Card):
        return NotImplemented
    return self.rank_int < other.rank_int

  def __eq__(self, other) -> bool:
    """Cardクラスのインスタンス同士が等しいかどうかを比較するためのメソッド\n
    スートに関係なく、ランクが同じなら等しいとみなす\n
    例: `Card('s', 14) == Card('h', 14)` は `True` になる"""
    if not isinstance(other, Card):
        return NotImplemented
    return self.rank_int == other.rank_int
  
  def __hash__(self) -> int:
    """Cardオブジェクトをハッシュ可能にするためのメソッド"""
    return hash((self.suit, self.rank_int))

def create_deck() -> list[Card]:
    """52枚のトランプカードで構成されるデッキを生成する\nシャッフルはされていない"""
    deck: list[Card] = []
    for suit in SUITS_MAP.keys():
        for rank_int in RANKS_MAP.keys():
            deck.append(Card(suit, rank_int))
    return deck

# ハンドクラスを作成する
class Hand:
  """2枚のハンドカードを表すクラス"""
  def __init__(self, cards: tuple[Card, Card]):
    """cards: 2枚のCardクラスのインスタンスをタプルで渡す"""
    self.cards = cards

  def __str__(self) -> str:
    """Handクラスのインスタンスを文字列に変換するためのメソッド\n
    例: `str(Hand((Card('s', 14), Card('s', 2))))` は `sA s2` になる"""
    # カードをランクの降順でソートしてから文字列に変換する
    sorted_cards = sorted(self.cards, reverse=True)
    return ' '.join(str(card) for card in sorted_cards)

  def __repr__(self) -> str:
    """Handクラスのインスタンスを文字列で表現するためのメソッド\n
    例: `repr(Hand((Card('s', 14), Card('s', 2))))` は `Hand((Card('s', 14), Card('s', 2)))` になる"""
    return f'Hand(({repr(self.cards[0])}, {repr(self.cards[1])}))' # これはソートせず生の順番で表現
  
  def get_all_cards(self) -> set[Card]:
    """ハンドカードをsetで返す"""
    return set(self.cards)
  
class Board:
  """ボードカードを表すクラス\n
  flopsは3枚、turnは1枚、riverは1枚のカードを持つ"""
  # flops, turn, riverはすべてCardクラスのインスタンスを持つ
  # init状態ではすべてNoneで初期化され、あとからflops, turn, riverをセットする
  def __init__(self):
    self.flops: tuple[Card, Card, Card] | None = None
    self.turn: Card | None = None
    self.river: Card | None = None

  def set_flops(self, cards: tuple[Card, Card, Card]):
    """flopsをセットするためのメソッド"""
    self.flops = cards

  def set_turn(self, card: Card):
    """turnをセットするためのメソッド"""
    self.turn = card

  def set_river(self, card: Card):
    """riverをセットするためのメソッド"""
    self.river = card

  def __str__(self) -> str:
    """Boardクラスのインスタンスを文字列に変換するためのメソッド\n
    例: str(Board())は`flops: None, turn: None, river: None`になる\n
    オープンされた場合、`flops: sA sK sQ, turn: sJ, river: sT`のようになる"""
    flops_str = ' '.join(str(card) for card in self.flops) if self.flops else 'None'
    turn_str = str(self.turn) if self.turn else 'None'
    river_str = str(self.river) if self.river else 'None'
    return f'flops: {flops_str}, turn: {turn_str}, river: {river_str}'

  def __repr__(self) -> str:
    """Boardクラスのインスタンスを文字列で表現するためのメソッド\n
    例: repr(Board())は`Board(flops=None, turn=None, river=None)`になる\n
    オープンされた場合、`Board(flops=(Card('s', 14), Card('s', 13), Card('s', 12)), turn=Card('s', 11), river=Card('s', 10))`のようになる"""
    flops_repr = f'({", ".join(repr(card) for card in self.flops)})' if self.flops else 'None'
    turn_repr = repr(self.turn) if self.turn else 'None'
    river_repr = repr(self.river) if self.river else 'None'
    return f'Board(flops={flops_repr}, turn={turn_repr}, river={river_repr})'
  
  def get_all_cards(self) -> set[Card]:
    """ボードカードをすべてsetで返す"""
    cards: set[Card] = set()
    if self.flops:
      cards.update(self.flops)
    if self.turn:
      cards.add(self.turn)
    if self.river:
      cards.add(self.river)
    return cards
  
  def is_complete(self) -> bool:
    """ボードカードがすべてセットされているかどうかを返す"""
    if self.flops is None:
        return False
    if self.turn is None:
        return False
    if self.river is None:
        return False
    return True
