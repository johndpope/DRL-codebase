# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

from agent import *
from component import *
from network import *
from utils import *
from exp import *

import torch
import torch.nn.functional as F

import argparse
import pickle

def dqn_cartpole(config):
    config.task_fn = lambda: ClassicalControl('CartPole-v0', seed=0)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001)
    config.network_fn = lambda: FCNet([4, 128, 2])
    config.replay_fn = lambda: Replay(capacity=10000)
    config.target_network_update_freq = 1000
    config.loss_fn = F.smooth_l1_loss
    config.eps_scheduler = ExponentialEpsScheduler(eps_start=0.95, eps_end=0.05, eps_decay=200)
    config.policy_fn = lambda: GreedyPolicy(config.eps_scheduler)
    config.batch_size = 64
    config.double_q = False

    config.tag = 'baseline'
    config.out_dir = 'data'
    config.logger = Logger()
    config.print_every = 10
    config.save_every = 40

    agent = DQNAgent(config)
    run_episodes(agent)
    run_render(agent)


def dqn_mountaincar(config):
    config.task_fn = lambda: ClassicalControl('MountainCar-v0', seed=0, initial_running_reward=-200)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001)
    config.network_fn = lambda: FCNet([2, 128, 3])
    config.replay_fn = lambda: Replay(capacity=10000)
    config.target_network_update_freq = 1000
    config.loss_fn = F.smooth_l1_loss
    config.eps_scheduler = ExponentialEpsScheduler(eps_start=0.95, eps_end=0.05, eps_decay=200)
    config.policy_fn = lambda: GreedyPolicy(config.eps_scheduler)
    config.batch_size = 64
    config.double_q = False

    config.tag = 'baseline'
    config.out_dir = 'data'
    config.logger = Logger()
    config.print_every = 10
    config.save_every = 40

    agent = DQNAgent(config)
    run_episodes(agent)
    run_render(agent)


def reinforce_cartpole(config):
    config.task_fn = lambda: ClassicalControl('CartPole-v0', seed=0)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.01)
    config.network_fn = lambda: REINFORCEFCNet([4, 128, 2])
    config.entropy_weight = 0

    config.tag = 'baseline'
    config.out_dir = 'data'
    config.logger = Logger()
    config.print_every = 10
    config.save_every = 40
    # config.logger.setLevel(logging.DEBUG)

    agent = REINFORCEAgent(config)
    run_episodes(agent)
    run_render(agent)


def reinforce_mountaincar(config):
    config.task_fn = lambda: ClassicalControl('MountainCar-v0', seed=0, initial_running_reward=-200)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.01)
    config.network_fn = lambda: REINFORCEFCNet([2, 128, 3])
    config.entropy_weight = 0

    config.tag = 'baseline'
    config.out_dir = 'data'
    config.logger = Logger()
    config.print_every = 10
    config.save_every = 40
    # config.logger.setLevel(logging.DEBUG)

    agent = REINFORCEAgent(config)
    run_episodes(REINFORCEAgent(config))
    run_render(agent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=('train', 'experiment'), default='train',
                        help='run: single run train; exp: run experiment')
    parser.add_argument('--agent', type=str, choices=('REINFORCE', 'DQN'), default='REINFORCE')
    parser.add_argument('--task', type=str, choices=('cartpole', 'mountaincar'), default='cartpole')
    parser.add_argument('--exp_id', type=int)
    parser.add_argument('--space_file', type=str)
    parser.add_argument('--out_dir', type=str, default='data/')
    parser.add_argument('--num_runs', type=int)
    args = parser.parse_args()

    if args.mode == 'train':
        config = Config()
        config.out_dir = args.out_dir
        if args.agent == 'REINFORCE':
            if args.task == 'cartpole':
                reinforce_cartpole(config)
            elif args.task == 'mountaincar':
                reinforce_mountaincar(config)
        elif args.agent == 'DQN':
            if args.task == 'cartpole':
                dqn_cartpole(config)
            elif args.task == 'mountaincar':
                dqn_mountaincar(config)

    elif args.mode == 'experiment':
        with open(args.space_file, 'rb') as f:
            space = pickle.load(f)
        exp = ExpSuite(args.agent, args.task)
        exp.run_exp(space, args.out_dir, args.exp_id, args.num_runs)


if __name__ == '__main__':
    main()