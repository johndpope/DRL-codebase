# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

class Config(object):
    def __init__(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.network_fn = None
        self.policy_fn = None
        self.replay_fn = None
        self.gamma = 0.99
        self.logger = None
        self.print_every = None
        self.save_every = None
        self.tag = None

        # DQN
        self.hidden_dims = None

        self.capacity = None
        self.batch_size = None

        self.target_network_update_freq = None

        self.optimizer = None
        self.optimizer_params = None

        self.loss_fn = None

        self.double_q = None

        # REINFORCE
        self.entropy_weight = None



