# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

class QFunctions(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=81):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels,n_hidden_channels),
            l2=L.Linear(n_hidden_channels,n_hidden_channels),
            l3=L.Linear(n_hidden_channels,n_actions)
        )

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))

