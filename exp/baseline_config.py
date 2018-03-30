# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

from component import *
from network import *
from utils import *
import torch
import torch.nn.functional as F


class DQNCartPoleConfig(Config):
    def __init__(self):
        super(DQNCartPoleConfig, self).__init__()

        self.optimizer = torch.optim.RMSprop
        self.lr = 0.001

        self.hidden_dims = (128,)
        self.target_network_update_freq = 1000

        self.loss_fn = F.smooth_l1_loss

        self.batch_size = 64

        self.double_q = False

        self.print_every = 10
        self.save_every = 40

    def set(self):
        self.task_fn = lambda: ClassicalControl('CartPole-v0', seed=self.task_seed)
        self.optimizer_fn = lambda params: self.optimizer(params, lr=self.lr)
        self.network_fn = lambda: FCNet((4,) + self.hidden_dims + (2,))
        self.replay_fn = lambda: Replay(capacity=self.capacity, batch_size=self.batch_size)
        self.eps_scheduler = ExponentialEpsScheduler(eps_start=self.eps_start, eps_end=self.eps_end, eps_decay=self.eps_decay)
        self.policy_fn = lambda: GreedyPolicy(self.eps_scheduler)
        self.logger = Logger()


class REINFORCECartPoleConfig(Config):
    def __init__(self):
        super(REINFORCECartPoleConfig, self).__init__()

        self.optimizer = torch.optim.Adam
        self.lr = 0.001

        self.hidden_dims = (128,)

        self.entropy_weight = 0

        self.print_every = 10
        self.save_every = 40

    def set(self):
        self.task_fn = lambda: ClassicalControl('CartPole-v0', seed=self.task_seed)
        self.optimizer_fn = lambda params: self.optimizer(params, lr=self.lr)
        self.network_fn = lambda: REINFORCEFCNet((4,) + self.hidden_dims + (2,))
        self.logger = Logger()