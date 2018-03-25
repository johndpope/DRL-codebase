# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

from network import *
from component import *
from utils import *
import torch
from torch.distributions import Categorical
import numpy as np
import time

class REINFORCEAgent(object):
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.log_probs = []
        self.rewards = []
        self.entropy_loss = 0

    def select_action(self, state):
        state = Variable(torch.from_numpy(state).unsqueeze(0))
        probs = self.network(state)
        m = Categorical(probs=probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        self.entropy_loss += torch.dot(torch.log(probs), probs)
        return action.data[0]

    def update_network(self):
        R = 0
        rewards = []
        for r in self.rewards[::-1]:
            R = R * self.config.gamma + r
            rewards.insert(0, R)
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        log_probs = torch.cat(self.log_probs)
        policy_loss = torch.dot(-log_probs, Variable(rewards))

        total_loss = policy_loss + self.entropy_loss * self.config.entropy_weight
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        del self.log_probs[:]
        del self.rewards[:]
        self.entropy_loss = 0

    def episode(self):
        episode_start_time = time.time()
        state = self.task.reset()
        steps_done = 0
        total_reward = 0
        while True:
            action = self.select_action(state)
            next_state, reward, done, info = self.task.step(action)
            self.rewards.append(reward)
            steps_done += 1
            total_reward += reward
            state = next_state

            if done:
                break

        self.update_network()

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %.2f, time per step %.2f' %
                                 (steps_done, episode_time, episode_time / steps_done))

        return steps_done, total_reward

    def save(self, file_name):
        torch.save(self.network.state_dict(), file_name)
