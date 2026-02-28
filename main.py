import argparse
from colorama import Fore, Style
from player import HumanPlayer, ConservativeCpu, BalancedCpu, AggressiveCpu
from gto_cpu import GtoCpu
from learning_game import LearningGame as Game
import time

def run_match(num_matches=1, cpu_only=False, initial_chips=1000, silent=False):
    """
    指定された回数のマッチ（誰か一人が勝ち残るまで）を実行する。
    """
    win_counts = {}
    
    for i in range(num_matches):
        if not silent:
            print(f"\n{Fore.YELLOW}=== マッチ {i+1} / {num_matches} 開始 ==={Style.RESET_ALL}")
            
        # プレイヤーの作成
        players = []
        if not cpu_only:
            players.append(HumanPlayer("You", chips=initial_chips))
        
        # CPUプレイヤーのバリエーション
        players.append(ConservativeCpu("Conservative", chips=initial_chips))
        players.append(BalancedCpu("Balanced-1", chips=initial_chips))
        players.append(BalancedCpu("Balanced-2", chips=initial_chips))
        players.append(AggressiveCpu("Aggressive", chips=initial_chips))
        players.append(GtoCpu("GTO-CPU", chips=initial_chips))
        
        # 統計用初期化（初回のみ）
        for p in players:
            if p.name not in win_counts:
                win_counts[p.name] = 0
                
        game = Game(players, start_chips=initial_chips, sb=10, bb=20, silent=silent)
        
        round_count = 0
        while True:
            # ゲーム終了の判定（残り1人以下）
            active_players = [p for p in game.players if p.chips > 0]
            if len(active_players) < 2:
                if len(active_players) == 1:
                    winner_name = active_players[0].name
                    win_counts[winner_name] += 1
                    # サイレントモードでもマッチ終了通知は出す
                    print(f"{Fore.GREEN}[マッチ {i+1}] 終了: 勝者は {winner_name} です！{Style.RESET_ALL}")
                else:
                    print(f"[マッチ {i+1}] 終了: 勝者はいません（全員がバスト）。")
                break
                
            # 人間プレイヤーが生存している場合のインタラクション
            if not cpu_only:
                human_player = next((p for p in game.players if isinstance(p, HumanPlayer)), None)
                if human_player and human_player.chips > 0:
                    if not silent:
                        print(f"\n次のラウンド({round_count+1})を開始しますか？ (y/n)")
                        choice = input("> ").strip().lower()
                        if choice == 'n':
                            print("マッチを中断します。")
                            return win_counts
                elif not silent:
                    # 人間がバストした後は少し待機して自動進行
                    print(f"{Fore.RED}あなたはバストしました。自動進行します...{Style.RESET_ALL}")
                    time.sleep(0.1)
            
            # 1ラウンドのプレイ
            game.play_round()
            round_count += 1
            
            # 状況表示（サイレントモードでない場合）
            if not silent:
                print(f"\n--- ラウンド {round_count} 終了時のスタック ---")
                for p in game.players:
                    status_str = " (BUST)" if p.chips == 0 else ""
                    name_color = Fore.CYAN if isinstance(p, HumanPlayer) else Style.RESET_ALL
                    print(f"{name_color}{p.name:15}{Style.RESET_ALL}: {p.chips:5} chips{status_str}")

    return win_counts

def main():
    parser = argparse.ArgumentParser(description="テキサスホールデム CLI / シミュレーター")
    parser.add_argument("-n", "--num-matches", type=int, default=1, help="実行するマッチ数 (デフォルト: 1)")
    parser.add_argument("--cpu-only", action="store_true", help="人間プレイヤーを除外してCPUのみで対戦させる")
    parser.add_argument("--chips", type=int, default=1000, help="各プレイヤーの初期チップ数 (デフォルト: 1000)")
    parser.add_argument("-s", "--silent", action="store_true", help="ラウンドごとの詳細出力を抑制する")
    args = parser.parse_args()

    print(f"=== テキサスホールデム シミュレーション開始 ===")
    print(f"設定: マッチ数={args.num_matches}, CPUのみ={args.cpu_only}, 初期チップ={args.chips}")
    
    start_time = time.time()
    try:
        results = run_match(
            num_matches=args.num_matches, 
            cpu_only=args.cpu_only, 
            initial_chips=args.chips,
            silent=args.silent
        )
    except KeyboardInterrupt:
        print("\nシミュレーションを中断しました。")
        return

    end_time = time.time()

    # 結果の表示
    print("\n" + "="*40)
    print(f"      シミュレーション結果 (全 {sum(results.values())} マッチ)")
    print("="*40)
    
    total_matches = sum(results.values())
    if total_matches > 0:
        # 勝数が多い順にソート
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for name, wins in sorted_results:
            win_rate = (wins / total_matches) * 100
            print(f"{name:15}: {wins:4} 勝 ({win_rate:5.1f}%)")
    else:
        print("マッチが完了しませんでした。")
    
    print("="*40)
    print(f"合計所要時間: {end_time - start_time:.2f}秒")

if __name__ == "__main__":
    main()
