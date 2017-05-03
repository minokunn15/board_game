# -*- coding: utf-8 -*-

class RandomActor:
    """
    ランダムに対戦させる時に用いるクラス
    """

    def __init__(self,board):
        """
        :param board: bordクラスのインスタンス
        """
        self.board = board
        self.random_count = 0

    def random_action(self):
        self.random_count += 1
        return self.board.get_empty_pos()