# -*- coding: utf-8 -*-
import numpy as np

class Board:
    """
    3目並べの盤面に関するクラス
    """

    def reset(self):
        self.board = np.array([0]*9,dtype=np.float32)
        self.winner = None
        self.missed = False
        self.done = False

    def move(self,act,turn):
        if self.board[act] == 0:
            self.board[act] = turn
            self.check_winner()
        else:
            self.winner = turn*-1
            self.missed = True
            self.done = True

    def check_winner(self):
        win_conditions = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for cond in win_conditions:
            if self.board[cond[0]] == self.board[cond[1]] == self.board[cond[2]]:
                if self.board[cond[0]] != 0:
                    self.winner = self.board[cond[0]]
                    self.done = True
                    return
        if np.count_nonzero(self.board) == 9:
            self.winner = 0
            self.done = True

    def get_empty_pos(self):
        empties = np.where(self.board == 0)[0]
        if len(empties) > 0:
            return np.random.choice(empties)
        else:
            return 0

    def show(self):
        row = " {} | {} | {} "
        hr = "\n-----------\n"
        tempboard = []
        for i in self.board:
            if i == 1:
                tempboard.append("○")
            elif i == -1:
                tempboard.append("×")
            else:
                tempboard.append(" ")
        print((row + hr + row + hr + row).format(*tempboard))
