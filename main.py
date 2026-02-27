from colorama import Fore, Style
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
            print("\n=== ゲーム終了 ===")
            if len(active_players) == 1:
                print(f"勝者は {active_players[0].name} です！")
            else:
                print("勝者はいません（全員がバスト）。")
            break
            
        # 人間プレイヤーがまだチップを持っているか確認
        human_is_alive = any(isinstance(p, HumanPlayer) and p.chips > 0 for p in game.players)
        
        if human_is_alive:
            print("\n次のラウンドを開始しますか？ (y/n)")
            choice = input("> ").strip().lower()
            if choice == 'n':
                print("ゲームを終了します。")
                break
        else:
            # 人間がバストした後は自動で進行
            import time
            print(f"\n{Fore.YELLOW}--- あなたはバストしました。CPU同士の対戦を自動で続行します ---{Style.RESET_ALL}")
            time.sleep(0.5)  # 状況を確認できるよう少し待機
            
        # 1ラウンドのプレイ
        game.play_round()
        
        # プレイヤーの状態表示
        print("\n--- 現在のスタック ---")
        for p in game.players:
            status_str = " (バスト)" if p.chips == 0 else ""
            name_color = Fore.CYAN if isinstance(p, HumanPlayer) else Style.RESET_ALL
            print(f"{name_color}{p.name}{Style.RESET_ALL}: {p.chips}チップ{status_str}")

if __name__ == "__main__":
    main()
