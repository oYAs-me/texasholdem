import random
from typing import List
from card import Card, Hand, Board, create_deck
from player import Player
from hand_strength import evaluate_hand

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
        
        # 初期状態リセット
        for p in self.players:
            p.chips = start_chips

    def get_active_players(self) -> List[Player]:
        return [p for p in self.players if p.status in ('active', 'all-in')]
    
    def get_contesting_players(self) -> List[Player]:
        # フォールドしていないプレイヤー
        return [p for p in self.players if p.status != 'folded' and p.status != 'busted']

    def start_round(self):
        """ラウンドの初期化、ブラインドの支払い、カードの配布"""
        print("\n--- 新しいラウンドを開始します ---")
        self.deck = create_deck()
        random.shuffle(self.deck)
        self.board = Board()
        self.pot = 0
        self.current_bet = 0
        
        # プレイヤー状態のリセット
        for p in self.players:
            if p.chips > 0:
                p.status = 'active'
            else:
                p.status = 'busted'
            p.current_bet = 0
            p.hand = None
            
        contesting = self.get_contesting_players()
        if len(contesting) < 2:
            print("プレイヤーが足りません。ゲーム終了です。")
            return False

        sb_pos = (self.dealer_pos + 1) % len(self.players)
        bb_pos = (self.dealer_pos + 2) % len(self.players)
        
        # Bustedプレイヤーをスキップするロジックが必要だが、簡易化のためそのまま配置位置で判定
        while self.players[sb_pos].status == 'busted':
            sb_pos = (sb_pos + 1) % len(self.players)
        bb_pos = (sb_pos + 1) % len(self.players)
        while self.players[bb_pos].status == 'busted':
            bb_pos = (bb_pos + 1) % len(self.players)
            
        print(f"ディーラー: {self.players[self.dealer_pos].name}")
        
        # ブラインド支払い
        sb_amount = self.players[sb_pos].pay(self.small_blind)
        bb_amount = self.players[bb_pos].pay(self.big_blind)
        self.pot += (sb_amount + bb_amount)
        self.current_bet = self.big_blind
        print(f"{self.players[sb_pos].name} がSB {sb_amount} を支払いました。")
        print(f"{self.players[bb_pos].name} がBB {bb_amount} を支払いました。")

        # カード配布
        for p in contesting:
            card1 = self.deck.pop()
            card2 = self.deck.pop()
            p.hand = Hand((card1, card2))

        return True

    def play_round(self):
        """1ラウンドの進行全体"""
        if not self.start_round():
            return False
            
        # プリフロップ
        print("\n[プリフロップ]")
        start_idx = (self.dealer_pos + 3) % len(self.players) # BBの左隣から
        if not self.betting_round(start_idx):
            self.end_round_early()
            return True
            
        # フロップ
        print("\n[フロップ]")
        self.board.set_flops((self.deck.pop(), self.deck.pop(), self.deck.pop()))
        print(f"ボード: {Player.hand_output_format(self.board.get_all_cards())}")
        start_idx = (self.dealer_pos + 1) % len(self.players) # SBから
        if not self.betting_round(start_idx):
            self.end_round_early()
            return True

        # ターン
        print("\n[ターン]")
        self.board.set_turn(self.deck.pop())
        print(f"ボード: {Player.hand_output_format(self.board.get_all_cards())}")
        if not self.betting_round(start_idx):
            self.end_round_early()
            return True

        # リバー
        print("\n[リバー]")
        self.board.set_river(self.deck.pop())
        print(f"ボード: {Player.hand_output_format(self.board.get_all_cards())}")
        if not self.betting_round(start_idx):
            self.end_round_early()
            return True

        # ショーダウン
        self.showdown()
        return True

    def betting_round(self, start_idx: int) -> bool:
        """ベッティングラウンドのループ処理"""
        for p in self.players:
            p.round_bet = 0
        round_max_bet = 0
        if self.current_bet > 0 and len(self.board.get_all_cards()) == 0:
            round_max_bet = self.big_blind
            for p in self.players:
                p.round_bet = p.current_bet
                
        active_count = len([p for p in self.players if p.status == 'active'])
        if active_count < 2 and round_max_bet == 0:
            return True

        last_raiser = -1
        current_idx = start_idx
        
        while True:
            p = self.players[current_idx]
            if p.status == 'active':
                call_amount = round_max_bet - p.round_bet
                valid_actions = ['fold']
                if call_amount == 0:
                    valid_actions.append('check')
                else:
                    valid_actions.append('call')
                
                if p.chips > call_amount:
                    valid_actions.append('raise')
                    
                game_state = {
                    'board': self.board,
                    'pot': self.pot,
                    'call_amount': call_amount,
                    'min_raise': call_amount + self.big_blind,
                    'players': [{'name': p_other.name, 'chips': p_other.chips, 'status': p_other.status} for p_other in self.players]
                }
                
                if hasattr(p, 'get_action'):
                    action, amount = p.get_action(valid_actions, game_state)
                elif hasattr(p, 'decide_action'):
                    action, amount = p.decide_action(valid_actions, game_state)
                else:
                    action, amount = 'fold', 0

                print(f"[{p.name}] アクション: {action}", end="")
                if amount > 0:
                    print(f" {amount}")
                else:
                    print()

                if action == 'fold':
                    p.status = 'folded'
                elif action in ('call', 'check'):
                    pay_amt = p.pay(call_amount)
                    p.round_bet += pay_amt
                    self.pot += pay_amt
                elif action == 'raise':
                    pay_amt = p.pay(amount - p.round_bet)
                    p.round_bet += pay_amt
                    self.pot += pay_amt
                    round_max_bet = p.round_bet
                    last_raiser = current_idx
            
            if len(self.get_contesting_players()) == 1:
                return False

            next_idx = (current_idx + 1) % len(self.players)
            
            active_p = [p for p in self.players if p.status == 'active']
            if len(active_p) == 0:
                break
                
            if next_idx == last_raiser:
                break
            if last_raiser == -1 and next_idx == start_idx:
                if self.current_bet > 0 and len(self.board.get_all_cards()) == 0:
                    bb_pos = (self.dealer_pos + 2) % len(self.players)
                    if current_idx == bb_pos:
                        break
                else:
                    break
                    
            current_idx = next_idx
            
        return True

    def end_round_early(self):
        winners = self.get_contesting_players()
        if winners:
            winner = winners[0]
            print(f"\n{winner.name} の勝利！ ポット {self.pot} を獲得しました。")
            winner.receive_winnings(self.pot)
        self.move_dealer()

    def showdown(self):
        print("\n--- ショーダウン ---")
        contesting = self.get_contesting_players()
        evaluated_hands = []
        for p in contesting:
            if p.hand:
                ev_hand = evaluate_hand(p.hand, self.board)
                evaluated_hands.append((ev_hand, p))
                secondary_rank_str = f", {ev_hand.secondary_rank}" if ev_hand.secondary_rank else ""
                print(f"{p.name}: {p.hand_output_format(p.hand.cards)} -> {ev_hand.hand_type}({ev_hand.primary_rank}{secondary_rank_str})")

        evaluated_hands.sort(key=lambda x: x[0].value, reverse=True)
        best_value = evaluated_hands[0][0].value
        winners = [p for ev, p in evaluated_hands if ev.value == best_value]

        win_amount = self.pot // len(winners)
        for w in winners:
            print(f"{w.name} がポット {win_amount} を獲得しました！")
            w.receive_winnings(win_amount)

        self.move_dealer()

    def move_dealer(self):
        self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
        while self.players[self.dealer_pos].status == 'busted':
            self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
