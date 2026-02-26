from typing import Any
from card import Hand, Board, Card

class Player:
    def __init__(self, name: str, chips: int):
        self.name = name
        self.chips = chips
        self.hand: Hand | None = None
        self.current_bet: int = 0
        self.status: str = 'active'  # 'active', 'folded', 'all-in', 'busted'

    @staticmethod
    def hand_output_format(cards: set[Card] | tuple[Card, ...] | list[Card]) -> str:
        """カードのセットを見やすい文字列に変換するための静的メソッド\n
        例: `hand_output_format({Card('s', 14), Card('h', 10)})` は `♠A ♥ T` と記号に変換させる"""
        symbol_map = {'s': '♠ ', 'h': '♥ ', 'd': '♢ ', 'c': '♣ '}
        output = []
        for card in cards:
            output.append(f"{symbol_map[card.suit]}{card.rank_str}")
        return " ".join(output)
    
    def reset_for_new_round(self):
        self.hand = None
        self.current_bet = 0
        if self.chips > 0:
            self.status = 'active'
        else:
            self.status = 'busted'

    def pay(self, amount: int) -> int:
        """指定された額を支払う。チップが足りない場合はオールインになる。"""
        actual_pay = min(amount, self.chips)
        self.chips -= actual_pay
        self.current_bet += actual_pay
        if self.chips == 0:
            self.status = 'all-in'
        return actual_pay

    def fold(self):
        self.status = 'folded'

    def receive_winnings(self, amount: int):
        self.chips += amount

class HumanPlayer(Player):
    def get_action(self, valid_actions: list[str], game_state: dict[str, Any]) -> tuple[str, int]:
        call_amount = game_state['call_amount']
        min_raise = game_state.get('min_raise', call_amount * 2)
        board = game_state.get('board')
        
        print(f"\n--- {self.name} のターン ---")
        hand_str = f"手札: {self.hand_output_format(self.hand.cards) if self.hand else 'なし'}"
        board_cards = []
        if board:
            if board.flops: board_cards.extend(board.flops)
            if board.turn: board_cards.append(board.turn)
            if board.river: board_cards.append(board.river)
            
        if board_cards:
            board_str = self.hand_output_format(board_cards)
            print(f"{hand_str} | ボード: [{board_str}]")
        else:
            print(hand_str)
            
        print(f"チップ: {self.chips}, 必要なコール額: {call_amount}")
        
        # 表示用の選択肢を作成
        display_actions = []
        default_action = None
        if 'check' in valid_actions:
            display_actions.append('(c)heck')
            default_action = 'check'
        elif 'call' in valid_actions:
            display_actions.append('(c)all')
            default_action = 'call'
            
        if 'fold' in valid_actions:
            display_actions.append('(f)old')
        if 'raise' in valid_actions:
            display_actions.append('(r)aise')
        
        prompt = f"アクションを選択してください [{', '.join(display_actions)}] (Enterで{default_action}): "
        
        while True:
            action_input = input(prompt).strip().lower()
            
            # 入力とアクションの紐付け
            chosen_action = None
            if action_input == '':
                chosen_action = default_action
            elif action_input == 'f':
                chosen_action = 'fold'
            elif action_input == 'c':
                chosen_action = 'check' if 'check' in valid_actions else 'call'
            elif action_input == 'r':
                chosen_action = 'raise'
            elif action_input in valid_actions:
                chosen_action = action_input
            
            if chosen_action in valid_actions:
                if chosen_action == 'call':
                    return 'call', call_amount
                elif chosen_action == 'check':
                    return 'check', 0
                elif chosen_action == 'fold':
                    return 'fold', 0
                elif chosen_action == 'raise':
                    while True:
                        raise_amt = input(f"レイズ額を入力してください (最小: {min_raise}, オールイン: {self.chips}): ")
                        try:
                            amt = int(raise_amt)
                            if amt >= min_raise and amt <= self.chips:
                                return 'raise', amt
                            else:
                                print(f"無効な額です。{min_raise} 以上、{self.chips} 以下を入力してください。")
                        except ValueError:
                            print("数値を入力してください。")
            else:
                print("無効なアクションです。")

class CpuAgent(Player):
    def decide_action(self, valid_actions: list[str], game_state: dict[str, Any]) -> tuple[str, int]:
        # デフォルトの実装（サブクラスでオーバーライドされることを想定）
        if 'check' in valid_actions:
            return 'check', 0
        return 'fold', 0

class ConservativeCpu(CpuAgent):
    """保守的なCPU: 強い手があるときだけ参加する"""
    def decide_action(self, valid_actions: list[str], game_state: dict[str, Any]) -> tuple[str, int]:
        from hand_strength import evaluate_hand
        call_amount = game_state['call_amount']
        board = game_state['board']
        
        if self.hand and board:
            ev_hand = evaluate_hand(self.hand, board)
            # プリフロップ（ボードなし）の場合
            if not board.get_all_cards():
                # ハイカードAまたはペアならコール
                if any(c.rank_int >= 13 for c in self.hand.cards) or self.hand.cards[0].rank_int == self.hand.cards[1].rank_int:
                    return 'call', call_amount if 'call' in valid_actions else 0
            else:
                # ワンペア以上ならコール、ツーペア以上ならレイズも検討
                if ev_hand.hand_type_rank >= 2: # ツーペア以上
                    if 'raise' in valid_actions:
                        return 'raise', game_state['min_raise']
                    return 'call', call_amount if 'call' in valid_actions else 0
                if ev_hand.hand_type_rank >= 1: # ワンペア
                    return 'call', call_amount if 'call' in valid_actions else 0
        
        if call_amount == 0 and 'check' in valid_actions:
            return 'check', 0
        return 'fold', 0

class BalancedCpu(CpuAgent):
    """標準的なCPU: 状況に応じて柔軟に対応する"""
    def decide_action(self, valid_actions: list[str], game_state: dict[str, Any]) -> tuple[str, int]:
        from hand_strength import evaluate_hand
        import random
        call_amount = game_state['call_amount']
        board = game_state['board']
        
        if self.hand and board:
            ev_hand = evaluate_hand(self.hand, board)
            if not board.get_all_cards():
                # プリフロップ: ランクの合計が一定以上なら参加
                if sum(c.rank_int for c in self.hand.cards) >= 20 or self.hand.cards[0].rank_int == self.hand.cards[1].rank_int:
                    return 'call', call_amount if 'call' in valid_actions else 0
            else:
                # ポストフロップ
                if ev_hand.hand_type_rank >= 3: # スリーカード以上
                    if 'raise' in valid_actions:
                        return 'raise', game_state['min_raise'] + 20
                    return 'call', call_amount
                elif ev_hand.hand_type_rank >= 1: # ワンペア以上
                    if call_amount > self.chips * 0.3: # コール額がチップの30%を超えるなら慎重に
                        if ev_hand.hand_type_rank >= 2: return 'call', call_amount
                        return 'fold', 0
                    return 'call', call_amount if 'call' in valid_actions else 0
        
        if call_amount == 0 and 'check' in valid_actions:
            return 'check', 0
        # たまにブラフ
        if random.random() < 0.05 and 'call' in valid_actions:
            return 'call', call_amount
            
        return 'fold', 0

class AggressiveCpu(CpuAgent):
    """攻撃的なCPU: ブラフを多用し、積極的にレイズする"""
    def decide_action(self, valid_actions: list[str], game_state: dict[str, Any]) -> tuple[str, int]:
        from hand_strength import evaluate_hand
        import random
        call_amount = game_state['call_amount']
        board = game_state['board']
        
        if self.hand and board:
            ev_hand = evaluate_hand(self.hand, board)
            # プリフロップ
            if not board.get_all_cards():
                if sum(c.rank_int for c in self.hand.cards) >= 15 or random.random() < 0.3:
                    if 'raise' in valid_actions and random.random() < 0.4:
                        return 'raise', game_state['min_raise']
                    return 'call', call_amount if 'call' in valid_actions else 0
            else:
                # ポストフロップ
                if ev_hand.hand_type_rank >= 1 or random.random() < 0.2: # ワンペア以上、または20%でブラフ
                    if 'raise' in valid_actions and random.random() < 0.3:
                        return 'raise', game_state['min_raise'] + 40
                    return 'call', call_amount if 'call' in valid_actions else 0

        if call_amount == 0 and 'check' in valid_actions:
            return 'check', 0
        return 'fold', 0
