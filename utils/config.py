# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

class Config(object):
    def __init__(self):
        self.task_fn = None
        self.task_seed = None
        self.max_episodes = None
        self.gamma = 0.99

        self.optimizer_fn = None
        self.optimizer = None
        self.lr = None

        self.network_fn = None
        self.hidden_dims = None

        self.tag = None
        self.out_dir = None
        self.logger = None
        self.print_every = None
        self.save_every = None

        # DQN
        self.policy_fn = None
        self.eps_start = None
        self.eps_end = None
        self.eps_decay = None

        self.replay_fn = None
        self.capacity = None
        self.batch_size = None

        self.target_network_update_freq = None

        self.loss_fn = None

        self.double_q = None

        # REINFORCE
        self.entropy_weight = None

        


