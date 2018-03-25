# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

import numpy as np


class EpsScheduler(object):
    def __init__(self, eps_start):
        self.eps_start = eps_start

    def get_eps(self, step):
        raise NotImplementedError


class ExponentialEpsScheduler(EpsScheduler):
    def __init__(self, eps_decay, eps_start=0.95, eps_end=0.05):
        super(ExponentialEpsScheduler, self).__init__(eps_start)
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def get_eps(self, steps):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * steps / self.eps_decay)


class GreedyPolicy(object):
    def __init__(self, eps_scheduler):
        self.eps_scheduler = eps_scheduler
        self.steps_done = 0

    def sample(self, action_value):
        eps_threshold = self.eps_scheduler.get_eps(self.steps_done)
        self.steps_done += 1
        if np.random.rand() > eps_threshold:
            return np.argmax(action_value)
        else:
            return np.random.randint(0, len(action_value))

