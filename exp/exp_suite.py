# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

from exp import *
from agent import *
from utils import *
import os
import re
import itertools

class ExpSuite(object):
    def __init__(self, agent, task):
        if agent == 'REINFORCE':
            if task == 'cartpole':
                self.config = REINFORCECartPoleConfig()
                self.agent_fn = lambda: REINFORCEAgent(self.config)

        elif agent == 'DQN':
            if task == 'cartpole':
                self.config = DQNCartPoleConfig()
                self.agent_fn = lambda: DQNAgent(self.config)

    def run(self, num_seeds=3):
        for task_seed in range(num_seeds):
            self.config.task_seed = task_seed
            self.config.tag = update_seed_in_tag(self.config.tag, self.config.task_seed, 'task_seed')

            self.config.set()
            self.config.logger.info('Task seed %d' % self.config.task_seed)
            run_episodes(self.agent_fn())

    def run_exp(self, space, out_dir, id, num_seeds=10):
        self.config.out_dir = out_dir
        mkdir(out_dir)

        summary_dict = {}
        with open(os.path.join(out_dir, 'exp{}_summary.pkl'.format(id)), 'wb') as f:
            summary_dict['id'] = id
            summary_dict['space'] = serialize(space)
            summary_dict['num_seeds'] = num_seeds
            summary_dict['config'] = serialize(dict_config(self.config))
            pickle.dump(summary_dict, f)

        name_list = [x[0] for x in space.items()]
        vals_list = [x[1] for x in space.items()]
        for val_list in itertools.product(*vals_list):
            self.config.tag = 'exp{}'.format(id)
            for name, val in zip(name_list, val_list):
                setattr(self.config, name, val)
                self.config.tag += '-{}={}'.format(name, val)
            self.run(num_seeds)


