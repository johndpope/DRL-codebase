# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

import numpy as np

class Replay(object):
    def __init__(self, capacity, dtype=np.float32):
        self.capacity = capacity
        self.dtype = dtype

        self.states = None
        self.actions = np.empty(self.capacity, dtype=np.int)
        self.rewards = np.empty(self.capacity, dtype=dtype)
        self.next_states = None
        self.dones = np.empty(self.capacity, dtype=dtype) # dones type will be np.flaot32, because of the computation of q_next * (1 - dones)

        self.pos = 0
        self.full = False

    def feed(self, experience):
        state, action, reward, next_state, done = experience

        if self.states is None:
            self.states = np.empty((self.capacity,) + state.shape, dtype=self.dtype)
            self.next_states = np.empty((self.capacity,) + state.shape, dtype=self.dtype)

        self.states[self.pos][:] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos][:] = next_state
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.capacity:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        memory_size = self.capacity if self.full else self.pos
        sampled_indices = np.random.randint(0, memory_size, batch_size)
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.dones[sampled_indices]]

    def __len__(self):
        return self.capacity if self.full else self.pos

