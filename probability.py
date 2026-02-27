import random
import json
import os
import time
from typing import List, Optional, Dict
from itertools import combinations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from card import Card, Hand, Board, create_deck
from hand_strength import evaluate_hand

# 1つ1つの組み合わせを評価するワーカー関数（並列実行用）
def _evaluate_combination_batch(my_hand: Hand, combinations_chunk: List[tuple], 
                                current_flops: Optional[tuple], 
                                current_turn: Optional[Card], 
                                current_river: Optional[Card]) -> Counter:
    stats = Counter()
    sim_board = Board()
    
    # 固定されているボードカードをセット
    sim_board.flops = current_flops
    sim_board.turn = current_turn
    sim_board.river = current_river

    for extra_cards in combinations_chunk:
        # 足りないカードを埋める
        # 順番に flops -> turn -> river の空いているところに割り当てる
        ptr = 0
        if sim_board.flops is None:
            sim_board.flops = (extra_cards[ptr], extra_cards[ptr+1], extra_cards[ptr+2])
            ptr += 3
        if sim_board.turn is None:
            sim_board.turn = extra_cards[ptr]
            ptr += 1
        if sim_board.river is None:
            sim_board.river = extra_cards[ptr]
            ptr += 1
        
        res = evaluate_hand(my_hand, sim_board)
        stats[res.hand_type] += 1
        
        # 次のループのために、新しく埋めた箇所だけリセット
        if current_flops is None: sim_board.flops = None
        if current_turn is None: sim_board.turn = None
        if current_river is None: sim_board.river = None
        
    return stats

def calculate_hand_distribution(my_hand: Hand, board: Board, parallel: bool = True) -> Dict[str, float]:
    current_board_cards = list(board.get_all_cards())
    needed_count = 5 - len(current_board_cards)
    
    # プリフロップかつJSONが存在する場合
    if needed_count == 5:
        precomputed = _load_precomputed_preflop(my_hand)
        if precomputed:
            return precomputed

    if needed_count <= 0:
        final_hand = evaluate_hand(my_hand, board)
        return {final_hand.hand_type: 1.0}

    full_deck = create_deck()
    known_cards = list(my_hand.cards) + current_board_cards
    remaining_deck = [c for c in full_deck if c not in known_cards]
    
    all_combos = list(combinations(remaining_deck, needed_count))
    total_combinations = len(all_combos)
    
    # 現在のボード状態を保存
    c_flops, c_turn, c_river = board.flops, board.turn, board.river
    
    if parallel and total_combinations > 10000:
        num_workers = os.cpu_count() or 1
        chunk_size = total_combinations // (num_workers * 4) + 1
        chunks = [all_combos[i:i + chunk_size] for i in range(0, total_combinations, chunk_size)]
        
        combined_stats = Counter()
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_evaluate_combination_batch, my_hand, chunk, c_flops, c_turn, c_river) for chunk in chunks]
            for f in futures:
                combined_stats.update(f.result())
    else:
        combined_stats = _evaluate_combination_batch(my_hand, all_combos, c_flops, c_turn, c_river)
        
    return {hand_type: count / total_combinations for hand_type, count in combined_stats.items()}

def _get_preflop_key(my_hand: Hand) -> str:
    c1, c2 = sorted(my_hand.cards, key=lambda x: x.rank_int, reverse=True)
    is_suited = c1.suit == c2.suit
    return f"{c1.rank_int},{c2.rank_int},{is_suited}"

def _load_precomputed_preflop(my_hand: Hand) -> Optional[Dict[str, float]]:
    json_path = "preflop_distributions.json"
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            key = _get_preflop_key(my_hand)
            return data.get(key)
    except: return None

def calculate_equity(my_hand: Hand, board: Board, num_opponents: int, num_simulations: int = 1000) -> float:
    if num_opponents <= 0: return 1.0
    full_deck = create_deck()
    known_cards = list(my_hand.cards) + list(board.get_all_cards())
    remaining_deck = [c for c in full_deck if c not in known_cards]
    wins = 0.0
    for _ in range(num_simulations):
        current_deck = remaining_deck.copy()
        random.shuffle(current_deck)
        sim_board = Board()
        if board.flops: sim_board.set_flops(board.flops)
        else: sim_board.set_flops((current_deck.pop(), current_deck.pop(), current_deck.pop()))
        if board.turn: sim_board.set_turn(board.turn)
        else: sim_board.set_turn(current_deck.pop())
        if board.river: sim_board.set_river(board.river)
        else: sim_board.set_river(current_deck.pop())
        my_score = evaluate_hand(my_hand, sim_board).value
        opponent_scores = []
        for _ in range(num_opponents):
            opp_hand = Hand((current_deck.pop(), current_deck.pop()))
            opp_score = evaluate_hand(opp_hand, sim_board).value
            opponent_scores.append(opp_score)
        max_opp_score = max(opponent_scores)
        if my_score > max_opp_score: wins += 1.0
        elif my_score == max_opp_score:
            num_winners = opponent_scores.count(max_opp_score) + 1
            wins += 1.0 / num_winners
    return wins / num_simulations

if __name__ == "__main__":
    from card import Card
    my_hand = Hand((Card('s', 14), Card('h', 14)))
    board = Board()
    print("Preflop (JSON/Parallel) Test...")
    dist = calculate_hand_distribution(my_hand, board)
    print(f"AA Preflop: {dist.get('TWO_PAIR', 0):.2%}")
    
    print("\nFlop Test (sA sK, board s2 s3 c4)...")
    flop_board = Board()
    flop_board.set_flops((Card('s', 2), Card('s', 3), Card('c', 4)))
    dist_flop = calculate_hand_distribution(Hand((Card('s', 14), Card('s', 13))), flop_board)
    print(f"Flush probability: {dist_flop.get('FLUSH', 0):.2%}")
