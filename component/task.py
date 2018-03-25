# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

import gym
import numpy as np
import sys

class BasicTask(object):
    def __init__(self, max_steps=sys.maxsize, dtype=np.float32):
        self.steps = 0
        self.max_steps = max_steps
        self.dtype = dtype

    def reset(self):
        self.steps = 0
        state = self.env.reset()
        return state.astype(self.dtype)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.steps += 1
        done = (done or self.steps >= self.max_steps)
        return next_state.astype(self.dtype), reward, done, info

    def random_action(self):
        return self.env.action_space.sample()


class ClassicalControl(BasicTask):
    def __init__(self, name, seed=None, max_steps=sys.maxsize):
        BasicTask.__init__(self, max_steps)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.success_threshold = self.env.spec.reward_threshold
        if seed:
            self.env.seed(seed)