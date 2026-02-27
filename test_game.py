import unittest
from game import Game
from player import Player

class MockPlayer(Player):
    def __init__(self, name, chips, actions):
        super().__init__(name, chips)
        self.actions = actions
        self.action_index = 0

    def decide_action(self, valid_actions, game_state):
        if self.action_index < len(self.actions):
            action_data = self.actions[self.action_index]
            self.action_index += 1
            return action_data
        return 'fold', 0
    
    def get_action(self, valid_actions, game_state):
        return self.decide_action(valid_actions, game_state)

class TestGameBetting(unittest.TestCase):
    def test_min_raise_enforcement(self):
        """不正に低いレイズ額が入力されても、最小レイズ額が維持されることを確認"""
        p1 = MockPlayer("You", 1000, [('raise', 60)])
        p2 = MockPlayer("CPU", 1000, [('raise', 40)]) # 不正な額（現在の60より低い）
        
        game = Game([p1, p2], start_chips=1000, sb=10, bb=20)
        # プリフロップ開始状態を擬似的に作る
        for p in game.players:
            p.status = 'active'
            p.round_bet = 0
        
        # p1(You)が60にレイズ
        # p2(CPU)が40にレイズ(誤り)を試みる
        
        # betting_roundのロジックの一部をトレース
        # 1. p1のアクション
        game.betting_round(start_idx=0)
        
        # p2の round_bet が 80 (60 + BB) になっているはず
        self.assertEqual(game.players[1].round_bet, 80)
        # p1の次の手番でのコール額がマイナスにならないことを確認
        # (p1.round_bet=60, round_max_bet=80 なので call_amount=20)
        
    def test_call_amount_calculation(self):
        """レイズ合戦の中でコール額が正しく計算されるか"""
        p1 = MockPlayer("You", 1000, [('raise', 60), ('call', 20)])
        p2 = MockPlayer("CPU", 1000, [('raise', 100)])
        
        game = Game([p1, p2], start_chips=1000, sb=10, bb=20)
        for p in game.players: p.status = 'active'
        
        game.betting_round(start_idx=0)
        
        # p1: raise 60 -> p2: raise 100 -> p1: call (100-60=40のはずだが、
        # MockPlayerに渡されるcall_amountが正しいか)
        # このテストでは実行後のチップ残高などで間接的に確認
        self.assertEqual(p1.chips, 1000 - 100) # 合計100支払っているはず
        self.assertEqual(p2.chips, 1000 - 100)

if __name__ == '__main__':
    unittest.main()
