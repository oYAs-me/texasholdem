import unittest
from card import Card, Hand, Board
from hand_strength import evaluate_hand, EvaluatedHand, HAND_RANK_MAP

class TestHandStrength(unittest.TestCase):

    def setUp(self):
        self.evaluator = evaluate_hand

    def test_royal_flush(self):
        # Royal Flush: 5枚のカードがA, K, Q, J, Tで同じスート
        # Example: sA, sK, sQ, sJ, sT
        hand_cards = (Card('s', 14), Card('s', 13)) # sA, sK
        board_cards = (Card('s', 12), Card('s', 11), Card('s', 10), Card('d', 2), Card('c', 3)) # sQ, sJ, sT, d2, c3
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])
        
        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'ROYAL_FLUSH')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['ROYAL_FLUSH'])

    def test_straight_flush(self):
        # Straight Flush: 5枚のカードが連続しており、すべて同じスート
        # Example: s9, s8, s7, s6, s5
        hand_cards = (Card('s', 9), Card('s', 8)) # s9, s8
        board_cards = (Card('s', 7), Card('s', 6), Card('s', 5), Card('d', 2), Card('c', 3)) # s7, s6, s5, d2, c3
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'STRAIGHT_FLUSH')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['STRAIGHT_FLUSH'])
        self.assertEqual(best_hand.primary_rank, 9) # 9ハイストレートフラッシュ

    def test_four_of_a_kind(self):
        # Four of a Kind: 同じランクのカードが4枚
        # Example: sA, hA, dA, cA, sK
        hand_cards = (Card('s', 14), Card('h', 14)) # sA, hA
        board_cards = (Card('d', 14), Card('c', 14), Card('s', 13), Card('d', 2), Card('c', 3)) # dA, cA, sK, d2, c3
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'FOUR_OF_A_KIND')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['FOUR_OF_A_KIND'])
        self.assertEqual(best_hand.primary_rank, 14) # Aのフォーカード
        self.assertEqual(best_hand.kicker_ranks, (13,)) # キッカーはK

    def test_full_house(self):
        # Full House: あるランクのカードが3枚、別のランクのカードが2枚
        # Example: sA, hA, dA, sK, hK
        hand_cards = (Card('s', 14), Card('h', 14)) # sA, hA
        board_cards = (Card('d', 14), Card('s', 13), Card('h', 13), Card('d', 2), Card('c', 3)) # dA, sK, hK, d2, c3
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'FULL_HOUSE')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['FULL_HOUSE'])
        self.assertEqual(best_hand.primary_rank, 14) # Aが3枚
        self.assertEqual(best_hand.secondary_rank, 13) # Kが2枚

    def test_flush(self):
        # Flush: 5枚のカードがすべて同じスート
        # Example: sA, sK, s8, s5, s2
        hand_cards = (Card('s', 14), Card('s', 13)) # sA, sK
        board_cards = (Card('s', 8), Card('s', 5), Card('s', 2), Card('d', 7), Card('c', 6)) # s8, s5, s2, d7, c6
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'FLUSH')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['FLUSH'])
        self.assertEqual(best_hand.primary_rank, 14) # Aハイフラッシュ
        self.assertEqual(best_hand.kicker_ranks, (13, 8, 5, 2)) # 全てのキッカー

    def test_straight(self):
        # Straight: 5枚のカードが連続している
        # Example: sA, hK, dQ, cJ, sT
        hand_cards = (Card('s', 14), Card('h', 13)) # sA, hK
        board_cards = (Card('d', 12), Card('c', 11), Card('s', 10), Card('d', 2), Card('c', 3)) # dQ, cJ, sT, d2, c3
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'STRAIGHT')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['STRAIGHT'])
        self.assertEqual(best_hand.primary_rank, 14) # Aハイストレート

    def test_straight_low_ace(self):
        # Straight: A-2-3-4-5 (Aはローとして扱われる)
        # Example: s5, h4, d3, c2, sA
        hand_cards = (Card('s', 5), Card('h', 4)) # s5, h4
        board_cards = (Card('d', 3), Card('c', 2), Card('s', 14), Card('d', 7), Card('c', 8)) # d3, c2, sA, d7, c8
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'STRAIGHT')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['STRAIGHT'])
        self.assertEqual(best_hand.primary_rank, 5) # 5ハイストレート (A-5)

    def test_three_of_a_kind(self):
        # Three of a Kind: 同じランクのカードが3枚
        # Example: sA, hA, dA, sK, sQ
        hand_cards = (Card('s', 14), Card('h', 14)) # sA, hA
        board_cards = (Card('d', 14), Card('s', 13), Card('s', 12), Card('d', 2), Card('c', 3)) # dA, sK, sQ, d2, c3
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'THREE_OF_A_KIND')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['THREE_OF_A_KIND'])
        self.assertEqual(best_hand.primary_rank, 14) # Aのスリーカード
        self.assertEqual(best_hand.kicker_ranks, (13, 12)) # キッカーはK, Q

    def test_two_pair(self):
        # Two Pair: 2組のペア
        # Example: sA, hA, sK, hK, sQ
        hand_cards = (Card('s', 14), Card('h', 14)) # sA, hA
        board_cards = (Card('s', 13), Card('h', 13), Card('s', 12), Card('d', 2), Card('c', 3)) # sK, hK, sQ, d2, c3
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'TWO_PAIR')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['TWO_PAIR'])
        self.assertEqual(best_hand.primary_rank, 14) # Aペア
        self.assertEqual(best_hand.secondary_rank, 13) # Kペア
        self.assertEqual(best_hand.kicker_ranks, (12,)) # キッカーはQ

    def test_one_pair(self):
        # One Pair: 1組のペア
        # Example: sA, hA, sK, sQ, sJ
        hand_cards = (Card('s', 14), Card('h', 14)) # sA, hA
        board_cards = (Card('s', 13), Card('s', 12), Card('s', 11), Card('d', 2), Card('c', 3)) # sK, sQ, sJ, d2, c3
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'ONE_PAIR')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['ONE_PAIR'])
        self.assertEqual(best_hand.primary_rank, 14) # Aペア
        self.assertEqual(best_hand.kicker_ranks, (13, 12, 11)) # キッカーはK, Q, J

    def test_high_card(self):
        # High Card: どの役も成立しない場合
        # Example: sA, hK, dQ, c9, s7
        hand_cards = (Card('s', 14), Card('h', 13)) # sA, hK
        board_cards = (Card('d', 12), Card('c', 9), Card('s', 7), Card('d', 5), Card('c', 3)) # dQ, c9, s7, d5, c3
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'HIGH_CARD')
        self.assertEqual(best_hand.hand_type_rank, HAND_RANK_MAP['HIGH_CARD'])
        self.assertEqual(best_hand.primary_rank, 14) # Aハイ
        self.assertEqual(best_hand.kicker_ranks, (13, 12, 9, 7)) # キッカーはK, Q, 9, 7

    def test_hand_comparison(self):
        # Different hand types
        rf = EvaluatedHand('ROYAL_FLUSH')
        sf = EvaluatedHand('STRAIGHT_FLUSH', primary_rank=9)
        fk = EvaluatedHand('FOUR_OF_A_KIND', primary_rank=14, kicker_ranks=(13,))
        fh = EvaluatedHand('FULL_HOUSE', primary_rank=14, secondary_rank=13)
        fl = EvaluatedHand('FLUSH', primary_rank=14, kicker_ranks=(13, 12, 11, 10))
        st = EvaluatedHand('STRAIGHT', primary_rank=14)
        tk = EvaluatedHand('THREE_OF_A_KIND', primary_rank=14, kicker_ranks=(13, 12))
        tp = EvaluatedHand('TWO_PAIR', primary_rank=14, secondary_rank=13, kicker_ranks=(12,))
        op = EvaluatedHand('ONE_PAIR', primary_rank=14, kicker_ranks=(13, 12, 11))
        hc = EvaluatedHand('HIGH_CARD', primary_rank=14, kicker_ranks=(13, 12, 11, 10))

        self.assertTrue(rf > sf)
        self.assertTrue(sf > fk)
        self.assertTrue(fk > fh)
        self.assertTrue(fh > fl)
        self.assertTrue(fl > st)
        self.assertTrue(st > tk)
        self.assertTrue(tk > tp)
        self.assertTrue(tp > op)
        self.assertTrue(op > hc)

        # Same hand type, different primary rank
        st_a = EvaluatedHand('STRAIGHT', primary_rank=14)
        st_k = EvaluatedHand('STRAIGHT', primary_rank=13)
        self.assertTrue(st_a > st_k)

        # Same hand type, same primary rank, different secondary rank (for Full House, Two Pair)
        fh_a_k = EvaluatedHand('FULL_HOUSE', primary_rank=14, secondary_rank=13)
        fh_a_q = EvaluatedHand('FULL_HOUSE', primary_rank=14, secondary_rank=12)
        self.assertTrue(fh_a_k > fh_a_q)

        tp_a_k_q = EvaluatedHand('TWO_PAIR', primary_rank=14, secondary_rank=13, kicker_ranks=(12,))
        tp_a_k_j = EvaluatedHand('TWO_PAIR', primary_rank=14, secondary_rank=13, kicker_ranks=(11,))
        self.assertTrue(tp_a_k_q > tp_a_k_j)

        # Same hand type, same primary/secondary rank, different kicker
        fk_a_k = EvaluatedHand('FOUR_OF_A_KIND', primary_rank=14, kicker_ranks=(13,))
        fk_a_q = EvaluatedHand('FOUR_OF_A_KIND', primary_rank=14, kicker_ranks=(12,))
        self.assertTrue(fk_a_k > fk_a_q)

        tk_a_k_q = EvaluatedHand('THREE_OF_A_KIND', primary_rank=14, kicker_ranks=(13, 12))
        tk_a_k_j = EvaluatedHand('THREE_OF_A_KIND', primary_rank=14, kicker_ranks=(13, 11))
        self.assertTrue(tk_a_k_q > tk_a_k_j)

        op_a_k_q_j = EvaluatedHand('ONE_PAIR', primary_rank=14, kicker_ranks=(13, 12, 11))
        op_a_k_q_t = EvaluatedHand('ONE_PAIR', primary_rank=14, kicker_ranks=(13, 12, 10))
        self.assertTrue(op_a_k_q_j > op_a_k_q_t)

        hc_a_k_q_j_9 = EvaluatedHand('HIGH_CARD', primary_rank=14, kicker_ranks=(13, 12, 11, 9))
        hc_a_k_q_j_8 = EvaluatedHand('HIGH_CARD', primary_rank=14, kicker_ranks=(13, 12, 11, 8))
        self.assertTrue(hc_a_k_q_j_9 > hc_a_k_q_j_8)

    # --- Additional Tests for get_best_hand ---

    def test_full_house_on_board_vs_hand_full_house(self):
        # Board: A-A-A-K-K (Full House Aces full of Kings)
        # Hand: J-J (Two Pair, but board is better)
        hand_cards = (Card('s', 11), Card('h', 11)) # sJ, hJ
        board_cards = (Card('s', 14), Card('h', 14), Card('d', 14), Card('c', 13), Card('s', 13)) # sA, hA, dA, cK, sK
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'FULL_HOUSE')
        self.assertEqual(best_hand.primary_rank, 14) # A
        self.assertEqual(best_hand.secondary_rank, 13) # K

    def test_straight_on_board_vs_hand_straight(self):
        # Board: K-Q-J-T-9 (Straight K-high)
        # Hand: A-2 (High Card)
        hand_cards = (Card('s', 14), Card('h', 2)) # sA, h2
        board_cards = (Card('s', 13), Card('h', 12), Card('d', 11), Card('c', 10), Card('s', 9)) # sK, hQ, dJ, cT, s9
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'STRAIGHT')
        self.assertEqual(best_hand.primary_rank, 14) # A-high straight

        # Board: K-J-T-9-8. Hand: A-Q. Should make A-K-Q-J-T (A-high straight).
        hand_cards_2 = (Card('s', 14), Card('h', 12)) # sA, hQ
        board_cards_2 = (Card('d', 13), Card('c', 11), Card('s', 10), Card('h', 9), Card('d', 8)) # dK, cJ, sT, h9, d8
        hand_2 = Hand(hand_cards_2)
        board_2 = Board()
        board_2.set_flops((board_cards_2[0], board_cards_2[1], board_cards_2[2]))
        board_2.set_turn(board_cards_2[3])
        board_2.set_river(board_cards_2[4])

        best_hand_2 = self.evaluator(hand_2, board_2)
        self.assertEqual(best_hand_2.hand_type, 'STRAIGHT')
        self.assertEqual(best_hand_2.primary_rank, 14) # A-high straight (A, K, Q, J, T)

    def test_flush_on_board_vs_hand_flush(self):
        # Board: sA-sK-sQ-sJ-s9 (Flush sA-high)
        # Hand: s2-h3 (Flush sA-high is still best)
        hand_cards = (Card('s', 2), Card('h', 3)) # s2, h3
        board_cards = (Card('s', 14), Card('s', 13), Card('s', 12), Card('s', 11), Card('s', 9)) # sA, sK, sQ, sJ, s9
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'FLUSH')
        self.assertEqual(best_hand.primary_rank, 14) # A-high flush
        self.assertEqual(best_hand.kicker_ranks, (13, 12, 11, 9))

        # Board: sK-sQ-sJ-s9-s7 (Flush sK-high)
        # Hand: sA-h2 (Now hand makes sA-high flush)
        hand_cards_2 = (Card('s', 14), Card('h', 2)) # sA, h2
        board_cards_2 = (Card('s', 13), Card('s', 12), Card('s', 11), Card('s', 9), Card('s', 7)) # sK, sQ, sJ, s9, s7
        hand_2 = Hand(hand_cards_2)
        board_2 = Board()
        board_2.set_flops((board_cards_2[0], board_cards_2[1], board_cards_2[2]))
        board_2.set_turn(board_cards_2[3])
        board_2.set_river(board_cards_2[4])

        best_hand_2 = self.evaluator(hand_2, board_2)
        self.assertEqual(best_hand_2.hand_type, 'FLUSH')
        self.assertEqual(best_hand_2.primary_rank, 14) # A-high flush
        # Kickers for sA, sK, sQ, sJ, s9 should be (13, 12, 11, 9)
        # Board cards are sK(13),sQ(12),sJ(11),s9(9),s7(7)
        # Hand cards are sA(14),h2(2)
        # Available s-suited cards: sA(14),sK(13),sQ(12),sJ(11),s9(9),s7(7)
        # Best 5 s-suited cards: sA,sK,sQ,sJ,s9. Ranks: 14, 13, 12, 11, 9
        self.assertEqual(best_hand_2.kicker_ranks, (13, 12, 11, 9))

    def test_two_pair_over_board_two_pair(self):
        # Board: K-K-Q-Q-7 (Two Pair K, Q)
        # Hand: A-A (now makes Three Pair, choose A-A, K-K, Q kicker)
        hand_cards = (Card('s', 14), Card('h', 14)) # sA, hA
        board_cards = (Card('d', 13), Card('c', 13), Card('s', 12), Card('h', 12), Card('d', 7)) # dK, cK, sQ, hQ, d7
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'TWO_PAIR')
        self.assertEqual(best_hand.primary_rank, 14) # A-A
        self.assertEqual(best_hand.secondary_rank, 13) # K-K
        self.assertEqual(best_hand.kicker_ranks, (12,)) # Q kicker

    def test_three_pair_scenario(self):
        # Board: K-K-Q-Q-7
        # Hand: A-A
        # Best hand should be A-A, K-K, Q kicker (NOT A-A, Q-Q, K kicker or K-K, Q-Q, A kicker)
        hand_cards = (Card('s', 14), Card('h', 14)) # sA, hA
        board_cards = (Card('d', 13), Card('c', 13), Card('s', 12), Card('h', 12), Card('d', 7)) # dK, cK, sQ, hQ, d7
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'TWO_PAIR')
        self.assertEqual(best_hand.primary_rank, 14) # AA
        self.assertEqual(best_hand.secondary_rank, 13) # KK
        self.assertEqual(best_hand.kicker_ranks, (12,)) # Q

    def test_board_forms_strongest_hand(self):
        # Board: sA, sK, sQ, sJ, sT (Royal Flush)
        # Hand: d2, c3 (irrelevant)
        hand_cards = (Card('d', 2), Card('c', 3))
        board_cards = (Card('s', 14), Card('s', 13), Card('s', 12), Card('s', 11), Card('s', 10))
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'ROYAL_FLUSH')
    
    def test_complex_board_full_house_vs_four_of_a_kind(self):
        # Hand: sA, hA
        # Board: dA, cA, sK, hQ, dJ
        hand_cards = (Card('s', 14), Card('h', 14)) # sA, hA
        board_cards = (Card('d', 14), Card('c', 14), Card('s', 13), Card('h', 12), Card('d', 11)) # dA, cA, sK, hQ, dJ
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'FOUR_OF_A_KIND')
        self.assertEqual(best_hand.primary_rank, 14) # Four Aces
        self.assertEqual(best_hand.kicker_ranks, (13,)) # K kicker

    def test_multiple_flushes(self):
        # Board: sA-sK-sQ-hJ-d10
        # Hand: s2-s3 (sA-sK-sQ-s3-s2 flush)
        hand_cards = (Card('s', 2), Card('s', 3)) # s2, s3
        board_cards = (Card('s', 14), Card('s', 13), Card('s', 12), Card('h', 11), Card('d', 10)) # sA, sK, sQ, hJ, d10
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'FLUSH')
        self.assertEqual(best_hand.primary_rank, 14) # A-high flush
        self.assertEqual(best_hand.kicker_ranks, (13, 12, 3, 2))

    def test_straight_with_multiple_cards_forming_it(self):
        # Board: d8, c9, sT
        # Hand: sJ, hQ (making Q-high straight using 8,9,T,J,Q)
        # Other cards: sA, h2
        hand_cards = (Card('s', 11), Card('h', 12)) # sJ, hQ
        board_cards = (Card('d', 8), Card('c', 9), Card('s', 10), Card('s', 14), Card('h', 2)) # d8, c9, sT, sA, h2
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'STRAIGHT')
        self.assertEqual(best_hand.primary_rank, 12) # Q-high straight

    def test_full_house_kicker_tie_breaker(self):
        # Hand: K-K
        # Board: A-A-A-Q-J
        # Best hand: A-A-A-K-K (Aces full of Kings)
        hand_cards = (Card('s', 13), Card('h', 13)) # sK, hK
        board_cards = (Card('s', 14), Card('h', 14), Card('d', 14), Card('c', 12), Card('s', 11)) # sA, hA, dA, cQ, sJ
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'FULL_HOUSE')
        self.assertEqual(best_hand.primary_rank, 14) # A's
        self.assertEqual(best_hand.secondary_rank, 13) # K's

    def test_two_pair_different_kickers(self):
        # Hand: A-A
        # Board: K-K-Q-J-10
        # Best hand: A-A-K-K-Q (Q kicker)
        hand_cards = (Card('s', 14), Card('h', 14)) # sA, hA
        board_cards = (Card('s', 13), Card('h', 13), Card('d', 12), Card('c', 11), Card('s', 10)) # sK, hK, dQ, cJ, sT
        hand = Hand(hand_cards)
        board = Board()
        board.set_flops((board_cards[0], board_cards[1], board_cards[2]))
        board.set_turn(board_cards[3])
        board.set_river(board_cards[4])

        best_hand = self.evaluator(hand, board)
        self.assertEqual(best_hand.hand_type, 'STRAIGHT')
        self.assertEqual(best_hand.primary_rank, 14) # A-high Straight

if __name__ == '__main__':
    unittest.main()
