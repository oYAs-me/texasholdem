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
                if p.name == "You":
                    print(f"--- {p.name} の手番 (ポット: {self.pot}, 参加人数: {len(self.get_contesting_players())}, コール額: {call_amount}) ---")
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
                    'min_raise': round_max_bet + self.big_blind,
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
                    # レイズ額が現在の最大ベットより小さい場合は修正する
                    actual_raise_to = max(amount, round_max_bet + self.big_blind)
                    pay_amt = p.pay(actual_raise_to - p.round_bet)
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
                evaluated_hands.append({'ev': ev_hand, 'player': p})
                secondary_rank_str = f", {ev_hand.secondary_rank}" if ev_hand.secondary_rank else ""
                print(f"{p.name}: {p.hand_output_format(p.hand.cards)} -> {ev_hand.hand_type}({ev_hand.primary_rank}{secondary_rank_str})")

        # 役の強さでソート（降順）
        evaluated_hands.sort(key=lambda x: x['ev'].value, reverse=True)

        # 各プレイヤーがこのハンドでポットに入れた総額（サイドポット計算用）
        # フォールドしたプレイヤーの拠出金もポットに含まれているため保持しておく
        contributions = {p: p.current_bet for p in self.players}
        
        while self.pot > 0:
            # まだ取り分がある（拠出金が残っている）勝者を特定
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
            
            if not winners:
                # 役のある勝者が誰もいない場合（通常ありえないが安全のため）
                break

            # この勝者たちが現在のサイドポットから獲得できる「一人あたりの最大額」
            # 勝者の中で最も拠出額（の残り）が少ないプレイヤーの額が上限になる
            max_per_winner = min(contributions[w] for w in winners)
            
            # 全プレイヤー（フォールドした人も含む）から、max_per_winner 以下の分をポットから集める
            total_winnable = 0
            for p in self.players:
                can_take = min(contributions[p], max_per_winner)
                total_winnable += can_take
                contributions[p] -= can_take
                self.pot -= can_take
            
            # 勝者で分配
            share = total_winnable // len(winners)
            for w in winners:
                print(f"{w.name} がポット {share} を獲得しました！")
                w.receive_winnings(share)
            
            # 端数は最初の勝者に
            if total_winnable % len(winners) != 0:
                winners[0].receive_winnings(total_winnable % len(winners))

        self.move_dealer()

    def move_dealer(self):
        self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
        while self.players[self.dealer_pos].status == 'busted':
            self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
