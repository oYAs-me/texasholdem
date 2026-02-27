from player import HumanPlayer, ConservativeCpu, BalancedCpu, AggressiveCpu
from gto_cpu import GtoCpu
from learning_game import LearningGame as Game

def main():
    print("=== テキサスホールデム CLI ===")
    
    # プレイヤーの作成 (1人の人間と多様なCPU)
    players = []
    players.append(HumanPlayer("You", chips=1000))
    players.append(ConservativeCpu("CPU-1", chips=1000))
    players.append(BalancedCpu("CPU-2", chips=1000))
    players.append(AggressiveCpu("CPU-3", chips=1000))
    players.append(BalancedCpu("CPU-4", chips=1000))
    players.append(AggressiveCpu("CPU-5", chips=1000))
    players.append(GtoCpu("GTO-CPU", chips=1000))
        
    game = Game(players, start_chips=1000, sb=10, bb=20)
    
    while True:
        # ゲーム終了の判定
        active_players = [p for p in game.players if p.chips > 0]
        if len(active_players) < 2:
            print("=== ゲーム終了 ===")
            if len(active_players) == 1:
                print(f"勝者は {active_players[0].name} です！")
            else:
                print("勝者はいません（全員がバスト）。")
            break
            
        print("\n次のラウンドを開始しますか？ (y/n)")
        choice = input("> ").strip().lower()
        if choice == 'n':
            print("ゲームを終了します。")
            break
            
        # 1ラウンドのプレイ
        game.play_round()
        
        # プレイヤーの状態表示
        print("\n--- 現在のスタック ---")
        for p in game.players:
            print(f"{p.name}: {p.chips}チップ" + (" (バスト)" if p.chips == 0 else ""))

if __name__ == "__main__":
    main()
