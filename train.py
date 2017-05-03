# -*- coding: utf-8 -*-
from bord import Board
from random import RandomActor
from dqn import QFunctions
import chainer
import chainerrl

if __name__ == "__main__":
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
    agent_p2 = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_frequency=1,
        target_update_frequency=100)