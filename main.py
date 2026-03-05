import argparse
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from colorama import Fore, Style
from tqdm import tqdm

from player import HumanPlayer, StyledCpu
from gto_cpu import GtoCpu
from bayesian_cpu import BayesianCpu
from learning_game import LearningGame as Game

# Windows の cp932 ではスーツ記号が表示できないため UTF-8 に設定する
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# スタイル値 → 名前のラベル
_STYLE_LABELS = ["Tight", "Careful", "Balanced", "Loose", "Aggro"]

def _style_label(s: float) -> str:
    """スタイル値 0〜1 を 5 段階ラベルに変換する。"""
    idx = min(int(s * len(_STYLE_LABELS)), len(_STYLE_LABELS) - 1)
    return _STYLE_LABELS[idx]


def _make_cpu_players(num_players, initial_chips, num_simulations):
    """
    固定の BayesianCpu + グラデーション StyledCpu でプレイヤープールを生成する。

    StyledCpu の style 値は stratified sampling（均等区間内でランダム）で決定する。
    これにより各マッチで conservative〜aggressive の多様な対戦相手が生成される。
    """
    fixed = [
        BayesianCpu("Bayesian", chips=initial_chips),
        # GtoCpu("GTO-CPU", chips=initial_chips, num_simulations=num_simulations),
    ]
    num_styled = max(1, num_players - len(fixed))

    # Stratified sampling: [0,1] を num_styled 個の等幅区間に分割し、各区間内でランダムに選択
    styles = [
        random.uniform(i / num_styled, (i + 1) / num_styled)
        for i in range(num_styled)
    ]
    random.shuffle(styles)  # 対戦順（ブラインド位置）にも多様性を持たせる

    # 同一ラベルが複数出る場合に通し番号を付ける
    label_counts: dict[str, int] = {}
    styled: list[StyledCpu] = []
    for s in styles:
        label = _style_label(s)
        cnt = label_counts.get(label, 0)
        name = label if cnt == 0 else f"{label}-{cnt}"
        label_counts[label] = cnt + 1
        styled.append(StyledCpu(name, chips=initial_chips, style=s))

    return (fixed + styled)[:num_players]


def _play_one_match(args):
    """1マッチを実行して (winner_name, round_count) を返す。サブプロセスで動く。"""
    from game import Game as BaseGame  # LearningGame ではなく基底クラスを使い保存を避ける
    num_players, initial_chips, num_simulations = args
    players = _make_cpu_players(num_players, initial_chips, num_simulations)
    game = BaseGame(players, start_chips=initial_chips, sb=10, bb=20, silent=True)
    rounds = 0
    while True:
        active = [p for p in game.players if p.chips > 0]
        if len(active) < 2:
            return (active[0].name if active else None), rounds
        game.play_round()
        rounds += 1


def run_cpu_matches(num_matches, num_players, initial_chips, num_simulations, num_workers, silent=False):
    """
    cpu_only モードでマッチを実行する。
    silent=False: 逐次実行してゲームの進行（アクション・コミュニティカード）を表示する。
    silent=True:  並列実行して tqdm でマッチ単位の進捗のみ表示する。
    """
    win_counts = defaultdict(int)

    if not silent:
        # 非 silent: 逐次実行してゲーム出力をそのまま流す
        for i in range(num_matches):
            print(f"\n{Fore.YELLOW}=== マッチ {i+1} / {num_matches} ==={Style.RESET_ALL}")
            players = _make_cpu_players(num_players, initial_chips, num_simulations)
            game = Game(players, start_chips=initial_chips, sb=10, bb=20, silent=False)
            rounds = 0
            while True:
                active = [p for p in game.players if p.chips > 0]
                if len(active) < 2:
                    winner = active[0].name if active else None
                    break
                game.play_round()
                rounds += 1
            if winner:
                win_counts[winner] += 1
            print(f"{Fore.GREEN}勝者: {winner} ({rounds} rounds){Style.RESET_ALL}")
    else:
        # silent: 並列実行 + tqdm
        task_args = (num_players, initial_chips, num_simulations)
        workers = min(num_workers, num_matches)
        with tqdm(total=num_matches, desc="Matches", unit="match", file=sys.stdout) as pbar:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_play_one_match, task_args) for _ in range(num_matches)]
                for future in as_completed(futures):
                    winner, rounds = future.result()
                    if winner:
                        win_counts[winner] += 1
                    pbar.set_postfix(winner=winner or "-", rounds=rounds)
                    pbar.update(1)

    return dict(win_counts)


def run_human_match(num_players, initial_chips, num_simulations, silent):
    """人間参加モード: インタラクティブに 1 マッチを実行する。"""
    players = [HumanPlayer("You", chips=initial_chips)]
    players += _make_cpu_players(num_players - 1, initial_chips, num_simulations)
    game = Game(players, start_chips=initial_chips, sb=10, bb=20, silent=silent)

    rounds = 0
    while True:
        active = [p for p in game.players if p.chips > 0]
        if len(active) < 2:
            winner = active[0].name if active else None
            break

        human = next((p for p in game.players if isinstance(p, HumanPlayer)), None)
        if human and human.chips <= 0:
            print(f"{Fore.RED}あなたはバストしました。{Style.RESET_ALL}")
            break

        if not silent:
            ans = input(f"\n次のラウンド ({rounds + 1}) を開始しますか？ (y/n) > ").strip().lower()
            if ans == "n":
                print("中断しました。")
                return None, rounds

        game.play_round()
        rounds += 1

        if not silent:
            print(f"\n--- ラウンド {rounds} 終了 ---")
            for p in game.players:
                color = Fore.CYAN if isinstance(p, HumanPlayer) else Style.RESET_ALL
                status = " (BUST)" if p.chips == 0 else ""
                print(f"{color}{p.name:15}{Style.RESET_ALL}: {p.chips:5} chips{status}")

    return winner, rounds


def main():
    parser = argparse.ArgumentParser(description="テキサスホールデム CLI")
    parser.add_argument("-n", "--num-matches", type=int, default=1)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("-s", "--silent", action="store_true")
    parser.add_argument("--players", type=int, default=6, choices=range(2, 11), metavar="N")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, metavar="N",
                        help="並列マッチ数 (--cpu-only 時)")
    parser.add_argument("--pretrain", type=float, default=0.0, metavar="M",
                        help="事前学習を M 分間実行 (--cpu-only 推奨)")
    parser.add_argument("--sims", type=int, default=200, metavar="N",
                        help="GtoCpu のモンテカルロ試行数")
    args = parser.parse_args()

    print(f"=== テキサスホールデム ===")
    print(f"マッチ={args.num_matches}  CPU={args.cpu_only}  チップ={args.chips}  "
          f"プレイヤー={args.players}  workers={args.workers}")

    if args.pretrain > 0:
        from gto_selfplay import run_selfplay
        print(f"\n事前学習 {args.pretrain:.1f} 分 (workers={args.workers}) ...")
        run_selfplay(
            num_hands=None,
            num_players=min(args.players, 6),
            save_path="gto_strategy.json",
            num_simulations=args.sims,
            num_workers=args.workers,
            verbose=False,
            max_seconds=args.pretrain * 60,
            run_rare=False,
        )
        print("事前学習完了。\n")

    start = time.time()
    try:
        if args.cpu_only:
            results = run_cpu_matches(
                args.num_matches, args.players, args.chips, args.sims, args.workers,
                silent=args.silent,
            )
        else:
            winner, rounds = run_human_match(args.players, args.chips, args.sims, args.silent)
            results = {winner: 1} if winner else {}
    except KeyboardInterrupt:
        print("\n中断しました。")
        return

    elapsed = time.time() - start
    total = sum(results.values())
    print(f"\n{'='*40}")
    print(f"  結果 ({total} マッチ)  所要時間: {elapsed:.2f}秒")
    print(f"{'='*40}")
    for name, wins in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:15}: {wins:4} 勝 ({wins / total * 100:5.1f}%)")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
