# -*- coding: utf-8 -*-

from human import HumanPlayer
from bord import Board
import numpy as np
import chainerrl
from dqn import QFunctions
import chainer
from randomAct import RandomActor

human_player = HumanPlayer()
b = Board()
ra = RandomActor(b)


# 環境の次元数
obs_size = 9
# 行動の次元数
n_actions = 9
q_func = QFunctions(obs_size,n_actions)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# 報酬の割引率
gamma = 0.95
# epsilon-greedyを使ってたまに冒険
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=ra.random_action
)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
# agentの生成
agent_p1 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer, replay_start_size=500, update_frequency=1,
    target_update_frequency=100
)
agent_p1.load("result_20000")

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
