from typing import Any
from colorama import Fore, Style
from card import Hand, Board, Card
from probability import calculate_equity, calculate_hand_distribution

class Player:
    def __init__(self, name: str, chips: int):
        self.name = name
        self.chips = chips
        self.hand: Hand | None = None
        self.current_bet: int = 0
        self.status: str = 'active'

    @staticmethod
    def hand_output_format(cards: list[Card] | tuple[Card, ...] | set[Card]) -> str:
        """スートに色を付けた形式でカードを表示する"""
        return " ".join([c.colored_str() for c in cards])
    
    def reset_for_new_round(self):
        self.hand = None
        self.current_bet = 0
        if self.chips > 0: self.status = 'active'
        else: self.status = 'busted'

    def pay(self, amount: int) -> int:
        actual_pay = min(amount, self.chips)
        self.chips -= actual_pay
        self.current_bet += actual_pay
        if self.chips == 0: self.status = 'all-in'
        return actual_pay

    def fold(self): self.status = 'folded'
    def receive_winnings(self, amount: int): self.chips += amount

class HumanPlayer(Player):
    def get_action(self, valid_actions: list[str], game_state: dict[str, Any]) -> tuple[str, int]:
        call_amount = game_state['call_amount']
        min_raise = game_state.get('min_raise', call_amount * 2)
        board = game_state['board']
        pot = game_state['pot']
        num_opponents = len([p for p in game_state['players'] if p['status'] in ('active', 'all-in')]) - 1
        
        print(f"\n--- {Fore.CYAN}{self.name}{Style.RESET_ALL} のターン (ポット: {Fore.GREEN}{pot}{Style.RESET_ALL}) ---")
        if self.hand and board:
            dist = calculate_hand_distribution(self.hand, board)
            sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            dist_str = " | ".join([f"{k}: {v:.1%}" for k, v in sorted_dist[:4]])
            print(f"【完成確率】{dist_str}")
            equity = calculate_equity(self.hand, board, num_opponents, num_simulations=500)
            print(f"【推定勝率】{Fore.YELLOW}{equity:.1%}{Style.RESET_ALL} (vs {num_opponents}人)")

        hand_cards = sorted(list(self.hand.cards), reverse=True) if self.hand else []
        board_cards = []
        if board:
            if board.flops: board_cards.extend(board.flops)
            if board.turn: board_cards.append(board.turn)
            if board.river: board_cards.append(board.river)
        
        print(f"手札: [{self.hand_output_format(hand_cards)}] | ボード: [{self.hand_output_format(board_cards)}]")
            
        call_color = Fore.RED if call_amount > 0 else Style.RESET_ALL
        print(f"チップ: {self.chips}, 必要なコール額: {call_color}{call_amount}{Style.RESET_ALL}")
        
        display_actions = []
        default_action = None
        is_betting_phase = (call_amount == 0)
        if 'check' in valid_actions: display_actions.append('(c)heck'); default_action = 'check'
        elif 'call' in valid_actions: display_actions.append('(c)all'); default_action = 'call'
        if 'fold' in valid_actions: display_actions.append('(f)old')
        if 'raise' in valid_actions:
            display_actions.append('(b)et' if is_betting_phase else '(r)aise')
        
        prompt = f"アクション [{', '.join(display_actions)}] (Enterで{default_action}): "
        
        while True:
            action_input = input(prompt).strip().lower()
            chosen_action = None
            if action_input == '': chosen_action = default_action
            elif action_input == 'f': chosen_action = 'fold'
            elif action_input == 'c': chosen_action = 'check' if 'check' in valid_actions else 'call'
            elif action_input == 'r' or (is_betting_phase and action_input == 'b'): chosen_action = 'raise'
            
            if chosen_action in valid_actions:
                if chosen_action == 'call': return 'call', call_amount
                elif chosen_action == 'check': return 'check', 0
                elif chosen_action == 'fold': return 'fold', 0
                elif chosen_action == 'raise':
                    action_name = "ベット" if is_betting_phase else "レイズ"
                    while True:
                        print(f" (1/2ポット: {pot//2}, 2/3ポット: {int(pot*0.66)}, フルポット: {pot})")
                        raise_amt = input(f"{action_name}額 (最小: {min_raise}, オールイン: {self.chips}): ")
                        try:
                            amt = int(raise_amt)
                            if amt >= min_raise and amt <= self.chips: return 'raise', amt
                        except ValueError: pass
            print("無効なアクションです。")

class CpuAgent(Player):
    def decide_action(self, valid_actions: list[str], game_state: dict[str, Any]) -> tuple[str, int]:
        call_amount = game_state['call_amount']
        pot = game_state['pot']
        board = game_state['board']
        num_opponents = len([p for p in game_state['players'] if p['status'] in ('active', 'all-in')]) - 1
        equity = 0.5
        if self.hand and board:
            equity = calculate_equity(self.hand, board, num_opponents, num_simulations=400)
        if pot > 0 and call_amount > 0:
            bet_ratio = call_amount / pot
            if bet_ratio > 0.5: equity *= (1.0 - (min(bet_ratio, 1.5) * 0.2))
        pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 else 0
        return self._smart_action(equity, pot_odds, valid_actions, game_state)

    def _get_dynamic_raise_size(self, pot: int, min_raise: int, equity: float, personality_factor: float) -> int:
        base_size = pot
        if equity < 0.6: base_size = int(pot * 0.5)
        elif equity < 0.8: base_size = int(pot * 0.75)
        raise_to = int(base_size * personality_factor)
        rounded_raise = ((raise_to + 5) // 10) * 10
        return max(rounded_raise, min_raise)

class ConservativeCpu(CpuAgent):
    def _smart_action(self, equity: float, pot_odds: float, valid_actions: list[str], game_state: dict) -> tuple[str, int]:
        call_amount = game_state['call_amount']
        if equity > pot_odds + 0.15 or call_amount == 0:
            if equity > 0.8 and 'raise' in valid_actions:
                size = self._get_dynamic_raise_size(game_state['pot'], game_state['min_raise'], equity, 0.8)
                return 'raise', size
            return 'call' if 'call' in valid_actions else 'check', call_amount
        return 'fold', 0

class BalancedCpu(CpuAgent):
    def _smart_action(self, equity: float, pot_odds: float, valid_actions: list[str], game_state: dict) -> tuple[str, int]:
        call_amount = game_state['call_amount']
        if equity > pot_odds + 0.05 or call_amount == 0:
            if equity > 0.65 and 'raise' in valid_actions:
                size = self._get_dynamic_raise_size(game_state['pot'], game_state['min_raise'], equity, 1.0)
                return 'raise', size
            return 'call' if 'call' in valid_actions else 'check', call_amount
        return 'fold', 0

class AggressiveCpu(CpuAgent):
    def _smart_action(self, equity: float, pot_odds: float, valid_actions: list[str], game_state: dict) -> tuple[str, int]:
        import random
        call_amount = game_state['call_amount']
        if (equity > 0.55 or random.random() < 0.15) and 'raise' in valid_actions:
            size = self._get_dynamic_raise_size(game_state['pot'], game_state['min_raise'], equity, 1.2)
            return 'raise', size
        if equity > pot_odds - 0.05 or call_amount == 0 or random.random() < 0.1:
            return 'call' if 'call' in valid_actions else 'check', call_amount
        return 'fold', 0
