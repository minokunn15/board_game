# -*- coding: utf-8 -*-
from bord import Board
from randomAct import RandomActor
from dqn import QFunctions
import chainer
import chainerrl
import numpy as np

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

    # ここから学習スタート

    # 学習ゲーム回数
    n_episodes = 20000
    miss = 0
    win = 0
    draw = 0
    # 繰り返し実行
    for i in range(1, n_episodes+1):
        b.reset()
        reward = 0
        agents = [agent_p1,agent_p2]
        turn = np.random.choice([0,1])
        last_state = None
        while not b.done:
            action = agents[turn].act_and_train(b.board.copy(),reward)
            b.move(action,1)
            #ゲームが終わった場合
            if b.done == True:
                if b.winner == 1:
                    reward = 1
                    win += 1
                elif b.winner == 0:
                    draw += 1
                else:
                    reward = -1
                if b.missed is True:
                    miss += 1
                agents[turn].stop_episode_and_train(b.board.copy(), reward, True)
                if agents[1 if turn == 0 else 0].last_state is not None and b.missed is False:
                    agents[1 if turn == 0 else 0].stop_episode_and_train(last_state, reward*-1, True)
            else:
                last_state = b.board.copy()
                b.board = b.board * -1
                turn = 1 if turn == 0 else 0

                #コンソールに進捗表示
        if i % 100 == 0:
            print("episode:", i, " / rnd:", ra.random_count, " / miss:", miss, " / win:", win, " / draw:", draw, " / statistics:", agent_p1.get_statistics(), " / epsilon:", agent_p1.explorer.epsilon)
            #カウンタの初期化
            miss = 0
            win = 0
            draw = 0
            ra.random_count = 0
        if i % 10000 == 0:
            # 10000エピソードごとにモデルを保存
            agent_p1.save("result_" + str(i))

    print("Training finished.")