# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

import torch
from torch.autograd import Variable
import time


class DQNAgent(object):
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.learning_network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.learning_network.state_dict())
        self.optimizer = config.optimizer_fn(self.learning_network.parameters())
        self.criterion = config.loss_fn
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps_done = 0  # accumulated steps in all experienced episodes

    def select_action(self, state):
        state_var = Variable(torch.from_numpy(state), volatile=True)
        action_value = self.learning_network(state_var).data.numpy().flatten()
        return self.policy.sample(action_value)

    def update_learning_network(self):
        if len(self.replay) < self.config.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.config.batch_size)

        states_var = Variable(torch.from_numpy(states))
        actions_var = Variable(torch.from_numpy(actions).unsqueeze(1))
        rewards_var = Variable(torch.from_numpy(rewards))
        next_states_var = Variable(torch.from_numpy(next_states))
        dones_var = Variable(torch.from_numpy(dones))

        if self.config.double_q:
            _, best_actions = self.learning_network(next_states_var).max(1)
            best_actions = best_actions.unsqueeze(1)

            q_next = self.target_network(next_states_var).gather(1, best_actions)
            q_next = q_next.squeeze(1)

        else:
            q_next, _ = self.target_network(next_states_var).max(1)

        q_next = q_next * (1 - dones_var)
        q_expected = q_next * self.config.gamma + rewards_var

        q = self.learning_network(states_var).gather(1, actions_var)

        loss = self.criterion(q, q_expected.detach())
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.learning_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()


    def episode(self):
        episode_start_time = time.time()
        state = self.task.reset()
        steps_done = 0
        total_reward = 0
        while True:
            action = self.select_action(state)
            next_state, reward, done, info = self.task.step(action)
            steps_done += 1
            total_reward += reward
            self.total_steps_done += 1

            self.replay.feed([state, action, reward, next_state, int(done)])
            state = next_state

            self.update_learning_network()

            if self.total_steps_done % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.learning_network.state_dict())

            if done:
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %.2f, time per step %.2f' %
                                 (steps_done, episode_time, episode_start_time / steps_done))

        return steps_done, total_reward

    def save(self, file_name):
        torch.save(self.learning_network.state_dict(), file_name)
