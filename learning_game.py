"""
learning_game.py

Game のサブクラス。ラウンド終了後にチップ差分から勝敗を判定し、
on_round_end(won) メソッドを持つ全プレイヤーに自動通知する。

・GtoCpu に限らず、on_round_end を実装した任意のプレイヤーが学習できる
・game.py / player.py への変更は不要
・main.py では `Game` の代わりにこちらを使う
"""
from __future__ import annotations

from game import Game


class LearningGame(Game):
    """
    プレイヤー型に依存しない事後学習フック付きの Game。

    play_round() の前後でチップ残高を比較し、
    on_round_end(won: bool) メソッドを持つプレイヤーに結果を通知する。
    """

    def play_round(self) -> bool:
        # ラウンド前のチップを記録
        chips_before = {p: p.chips for p in self.players}

        result = super().play_round()

        # チップが増えたプレイヤーを「勝者」とみなして通知
        for p in self.players:
            if hasattr(p, 'on_round_end'):
                won = p.chips > chips_before[p]
                p.on_round_end(won)

        return result
