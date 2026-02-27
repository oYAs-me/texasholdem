import json
import time
import os
from tqdm import tqdm
from card import Card, Hand, Board
from probability import calculate_hand_distribution

def get_all_preflop_combinations():
    """全169パターンのスターティングハンドの組み合わせを生成する"""
    combos = []
    
    # 1. ポケットペア (13通り)
    for r in range(14, 1, -1):
        combos.append((r, r, False, f"{Card('s', r).rank_str}{Card('h', r).rank_str}"))
        
    # 2. スーテッド & オフスート (78 * 2 = 156通り)
    for r1 in range(14, 1, -1):
        for r2 in range(r1 - 1, 1, -1):
            # スーテッド
            combos.append((r1, r2, True, f"{Card('s', r1).rank_str}{Card('s', r2).rank_str}s"))
            # オフスート
            combos.append((r1, r2, False, f"{Card('s', r1).rank_str}{Card('h', r2).rank_str}o"))
            
    return combos

def precompute():
    # 既存のデータをロード（途中から再開可能にするため）
    json_path = "preflop_distributions.json"
    results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
        except:
            pass

    all_combos = get_all_preflop_combinations()
    
    # すでに計算済みのものを除外（必要なら）
    # todo = [c for c in all_combos if f"{c[0]},{c[1]},{c[2]}" not in results]
    todo = all_combos # 今回は一から全て計算し直す想定
    
    print(f"全{len(todo)}パターンのプリフロップ事前計算を開始します...")
    
    # tqdmで進捗を表示
    pbar = tqdm(todo, desc="Calculating Hands", unit="hand")
    
    for r1, r2, suited, name in pbar:
        key = f"{r1},{r2},{suited}"
        
        # 進行状況をpbarに表示
        pbar.set_postfix(hand=name)
        
        # ハンド作成
        c1 = Card('s', r1)
        c2 = Card('h' if not suited else 's', r2)
        hand = Hand((c1, c2))
        board = Board()
        
        # 計算実行 (parallel=True は probability.py 側でマルチプロセスを使用)
        dist = calculate_hand_distribution(hand, board, parallel=True)
        results[key] = dist
        
        # 10手ごとに途中保存（クラッシュ対策）
        if len(results) % 10 == 0:
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
                
    # 最終保存
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n完了！ {len(results)}パターンのデータを '{json_path}' に保存しました。")

if __name__ == "__main__":
    precompute()
