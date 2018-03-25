# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

from agent import *
from component import *
from network import *
from utils import *

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dqn_cartpole(config):
    if config.use_default:
        config.task_fn = lambda: ClassicalControl('CartPole-v0', seed=0)
        config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001)
        config.network_fn = lambda: FCNet([4, 128, 2])
        config.replay_fn = lambda: Replay(capacity=10000)
        config.target_network_update_freq = 1000
        config.loss_fn = F.smooth_l1_loss
        config.eps_scheduler = ExponentialEpsScheduler(eps_start=0.95, eps_end=0.05, eps_decay=200)
        config.policy_fn = lambda: GreedyPolicy(config.eps_scheduler)
        config.batch_size = 64
        config.logger = Logger('./log')
        config.print_every = 10
        config.save_every = 40
        config.double_q = False
        config.tag = 'baseline'

    else:
        config.optimizer_fn = lambda params: config.optimizer(params, **config.optimizer_params)
        config.network_fn = lambda: FCNet((4,) + config.hidden_dims + (2,))
        config.replay_fn = lambda: Replay(capacity=config.capacity, batch_size=config.batch_size)

    run_episodes(DQNAgent(config))


def reinforce_cartpole(config):
    if config.use_default:
        config.task_fn = lambda: ClassicalControl('CartPole-v0', seed=0)
        config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.01)
        config.network_fn = lambda: REINFORCEFCNet([4, 128, 2])

        config.entropy_weight = 0

        config.logger = Logger('./log')
        config.print_every = 10
        config.save_every = 40
        config.tag = 'baseline'
        # config.logger.setLevel(logging.DEBUG)

    else:
        pass

    run_episodes(REINFORCEAgent(config))


if __name__ == '__main__':
    config = Config()
    config.use_default = True
    #dqn_cartpole(config)
    reinforce_cartpole(config)
