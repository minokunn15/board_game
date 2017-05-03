# -*- coding: utf-8 -*-
from human import HumanPlayer
from bord import Board
import numpy as np
import chainerrl

human_player = HumanPlayer()
b = Board()
agent_p1 = chainerrl.agents.DoubleDQN.load("result_20000")

for i in range(10):
    b.reset()
    dqn_first = np.random.choice([True,False])
    while not b.done:
        #DQN
        if dqn_first or np.count_nonzero(b.board) > 0:
            b.show()
            action = agent_p1.act(b.board.copy())
            b.move(action,1)
            if b.done == True:
                if b.winner == 1:
                    print("DQN WIN")
                elif b.winner == 0:
                    print("DRAW")
                else:
                    print("DQN MISSED")
                agent_p1.stop_episode()
                continue
        #人間
        b.show()
        action = human_player.act(b.board.copy())
        b.move(action, -1)
        if b.done == True:
            if b.winner == -1:
                print("HUMAN Win")
            elif b.winner == 0:
                print("Draw")
            agent_p1.stop_episode()

print("Test finished.")
