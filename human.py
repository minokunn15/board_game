# -*- coding: utf-8 -*-

class HumanPlayer:
    """
    3目並べを実際に人間が打つインターフェース
    """
    def act(self,board):
        valid = False
        while not valid:
            try:
                act = input("Please enter 1-9: ")
                act = int(act)
                if act >= 1 and act <= 9 and board[act-1] == 0:
                    valid = True
                    return act-1
                else:
                    print("Invalid move")
            except Exception as e:
                print(act + "is invalid")



