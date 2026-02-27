import random
from typing import List
from card import Card, Hand, Board, create_deck
from player import Player
from hand_strength import evaluate_hand
from colorama import Fore, Style, init

# Windowsでの色表示を有効化
init(autoreset=True)

class Game:
    def __init__(self, players: List[Player], start_chips: int = 1000, sb: int = 10, bb: int = 20):
        self.players = players
        self.deck: List[Card] = []
        self.board = Board()
        self.pot: int = 0
        self.dealer_pos: int = 0
        self.small_blind = sb
        self.big_blind = bb
        self.current_bet: int = 0
        
        for p in self.players:
            p.chips = start_chips

    def get_active_players(self) -> List[Player]:
        return [p for p in self.players if p.status in ('active', 'all-in')]
    
    def get_contesting_players(self) -> List[Player]:
        return [p for p in self.players if p.status != 'folded' and p.status != 'busted']

    def start_round(self):
        print(f"\n{Fore.GREEN}=== 新しいラウンドを開始します ==={Style.RESET_ALL}")
        self.deck = create_deck()
        random.shuffle(self.deck)
        self.board = Board()
        self.pot = 0
        self.current_bet = 0
        
        for p in self.players:
            p.reset_for_new_round()
            
        contesting = self.get_contesting_players()
        if len(contesting) < 2:
            print("プレイヤーが足りません。ゲーム終了です。")
            return False

        sb_pos = (self.dealer_pos + 1) % len(self.players)
        while self.players[sb_pos].status == 'busted':
            sb_pos = (sb_pos + 1) % len(self.players)
        bb_pos = (sb_pos + 1) % len(self.players)
        while self.players[bb_pos].status == 'busted':
            bb_pos = (bb_pos + 1) % len(self.players)
            
        print(f"ディーラー: {Fore.CYAN}{self.players[self.dealer_pos].name}{Style.RESET_ALL}")
        
        # ブラインド支払い
        sb_amount = self.players[sb_pos].pay(self.small_blind)
        bb_amount = self.players[bb_pos].pay(self.big_blind)
        self.pot += (sb_amount + bb_amount)
        self.current_bet = self.big_blind
        print(f"{Fore.YELLOW}{self.players[sb_pos].name}{Style.RESET_ALL} がSB {sb_amount} を支払いました。")
        print(f"{Fore.YELLOW}{self.players[bb_pos].name}{Style.RESET_ALL} がBB {bb_amount} を支払いました。")

        for p in contesting:
            card1 = self.deck.pop()
            card2 = self.deck.pop()
            p.hand = Hand((card1, card2))
        return True

    def play_round(self):
        if not self.start_round(): return False
            
        # プリフロップ
        print(f"\n{Fore.BLUE}[プリフロップ]{Style.RESET_ALL}")
        start_idx = (self.dealer_pos + 3) % len(self.players)
        if not self.betting_round(start_idx):
            self.end_round_early()
            return True
            
        # 以降の各ストリート
        streets = [("フロップ", 3), ("ターン", 1), ("リバー", 1)]
        for name, count in streets:
            print(f"\n{Fore.BLUE}[{name}]{Style.RESET_ALL}")
            cards = [self.deck.pop() for _ in range(count)]
            if name == "フロップ": self.board.set_flops(tuple(cards))
            elif name == "ターン": self.board.set_turn(cards[0])
            elif name == "リバー": self.board.set_river(cards[0])
            
            print(f"ボード: [{Player.hand_output_format(self.board.get_all_cards())}]")
            start_idx = (self.dealer_pos + 1) % len(self.players)
            if not self.betting_round(start_idx):
                self.end_round_early()
                return True

        self.showdown()
        return True

    def betting_round(self, start_idx: int) -> bool:
        for p in self.players: p.round_bet = 0
        round_max_bet = 0
        if self.current_bet > 0 and len(self.board.get_all_cards()) == 0:
            round_max_bet = self.big_blind
            for p in self.players: p.round_bet = p.current_bet
                
        last_raiser = -1
        current_idx = start_idx
        
        while True:
            p = self.players[current_idx]
            if p.status == 'active':
                call_amount = round_max_bet - p.round_bet
                valid_actions = ['fold']
                if call_amount == 0: valid_actions.append('check')
                else: valid_actions.append('call')
                if p.chips > call_amount: valid_actions.append('raise')
                    
                game_state = {
                    'board': self.board, 'pot': self.pot, 'call_amount': call_amount,
                    'min_raise': round_max_bet + self.big_blind,
                    'players': [{'name': p_o.name, 'chips': p_o.chips, 'status': p_o.status} for p_o in self.players]
                }
                
                # アクション取得
                if hasattr(p, 'decide_action'): action, amount = p.decide_action(valid_actions, game_state)
                else: action, amount = p.get_action(valid_actions, game_state)

                # 表示ロジック
                name_fmt = f"{Fore.CYAN}{p.name:^7}{Style.RESET_ALL}"
                if action == 'fold':
                    print(f"[{name_fmt}] {Fore.RED}fold{Style.RESET_ALL}")
                    p.status = 'folded'
                elif action == 'check':
                    print(f"[{name_fmt}] check")
                    p.pay(0)
                elif action == 'call':
                    pay_amt = p.pay(call_amount)
                    p.round_bet += pay_amt
                    self.pot += pay_amt
                    print(f"[{name_fmt}] {Fore.BLUE}call{Style.RESET_ALL} (支払: {pay_amt})")
                elif action == 'raise':
                    actual_raise_to = max(amount, round_max_bet + self.big_blind)
                    increment = actual_raise_to - round_max_bet
                    pay_amt = p.pay(actual_raise_to - p.round_bet)
                    p.round_bet += pay_amt
                    self.pot += pay_amt
                    round_max_bet = p.round_bet
                    last_raiser = current_idx
                    action_label = "bet" if call_amount == 0 else "raise"
                    print(f"[{name_fmt}] {Fore.YELLOW}{action_label:5}{Style.RESET_ALL} {round_max_bet} (+{increment})")
            
            if len(self.get_contesting_players()) == 1: return False
            next_idx = (current_idx + 1) % len(self.players)
            if len([p for p in self.players if p.status == 'active']) == 0: break
            if next_idx == last_raiser: break
            if last_raiser == -1 and next_idx == start_idx:
                if self.current_bet > 0 and len(self.board.get_all_cards()) == 0:
                    bb_pos = (self.dealer_pos + 2) % len(self.players)
                    if current_idx == bb_pos: break
                else: break
            current_idx = next_idx
        return True

    def end_round_early(self):
        winners = self.get_contesting_players()
        if winners:
            winner = winners[0]
            print(f"\n{Fore.GREEN}{winner.name} の勝利！ ポット {self.pot} を獲得しました。{Style.RESET_ALL}")
            winner.receive_winnings(self.pot)
        self.move_dealer()

    def showdown(self):
        print(f"\n{Fore.MAGENTA}--- ショーダウン ---{Style.RESET_ALL}")
        contesting = self.get_contesting_players()
        evaluated_hands = []
        for p in contesting:
            if p.hand:
                ev_hand = evaluate_hand(p.hand, self.board)
                evaluated_hands.append({'ev': ev_hand, 'player': p})
                
                # 手札とベスト5枚の表示
                h_str = Player.hand_output_format(p.hand.cards)
                best_str = Player.hand_output_format(ev_hand.best_cards)
                print(f"{Fore.CYAN}{p.name:^7}{Style.RESET_ALL}: [{h_str}]")
                print(f"  -> {Fore.MAGENTA}{ev_hand.hand_type:15}{Style.RESET_ALL} [{best_str}]")

        # 役の強さでソート
        evaluated_hands.sort(key=lambda x: x['ev'].value, reverse=True)
        contributions = {p: p.current_bet for p in self.players}
        
        while self.pot > 0:
            best_val = -1
            winners = []
            for item in evaluated_hands:
                p = item['player']
                if contributions[p] > 0:
                    if best_val == -1:
                        best_val = item['ev'].value
                        winners.append(p)
                    elif item['ev'].value == best_val:
                        winners.append(p)
            if not winners: break

            max_per_winner = min(contributions[w] for w in winners)
            total_winnable = 0
            for p in self.players:
                can_take = min(contributions[p], max_per_winner)
                total_winnable += can_take
                contributions[p] -= can_take
                self.pot -= can_take
            
            share = total_winnable // len(winners)
            for w in winners:
                print(f"{Fore.GREEN}{w.name} がポット {share} を獲得しました！{Style.RESET_ALL}")
                w.receive_winnings(share)
            if total_winnable % len(winners) != 0:
                winners[0].receive_winnings(total_winnable % len(winners))

        self.move_dealer()

    def move_dealer(self):
        self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
        while self.players[self.dealer_pos].status == 'busted':
            self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
