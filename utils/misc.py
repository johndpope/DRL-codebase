# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

import collections
import pickle
import os
import re


def run_episodes(agent):
    episodes_done = 0
    running_reward = agent.task.initial_running_reward
    stats = collections.defaultdict(lambda: [])
    while True:
        episodes_done += 1
        steps_done, reward = agent.episode()
        stats['steps_done'].append(steps_done)
        stats['reward'].append(reward)
        running_reward = running_reward * 0.99 + reward * 0.01

        if episodes_done % agent.config.print_every == 0:
            agent.config.logger.info('episode %d, reward %.2f, running reward %.2f' % (episodes_done, reward, running_reward))

        if agent.config.save_every and episodes_done % agent.config.save_every == 0:
            with open('%s/%s-%s-online-stats-%s.pkl' % (agent.config.out_dir, agent.__class__.__name__, agent.config.tag, agent.task.name), 'wb') as f:
                pickle.dump(dict(stats), f)

        if running_reward > agent.task.success_threshold:
            agent.config.logger.info('Solved! Running reward %.2f, last episode reward %d' % (running_reward, reward))
            break

        if agent.config.max_episodes and episodes_done > agent.config.max_episodes:
            break

    agent.save('%s/%s-%s-model-%s.pkl' % (agent.config.out_dir, agent.__class__.__name__, agent.config.tag, agent.task.name))
    with open('%s/%s-%s-all-stats-%s.pkl' % (agent.config.out_dir, agent.__class__.__name__, agent.config.tag, agent.task.name), 'wb') as f:
        pickle.dump(dict(stats), f)
    os.remove('%s/%s-%s-online-stats-%s.pkl' % (agent.config.out_dir, agent.__class__.__name__, agent.config.tag, agent.task.name))


def run_render(agent, num_episodes=5):
    for i in range(num_episodes):
        state = agent.task.reset()
        while True:
            action = agent.select_action(state)
            agent.task.render()
            next_state, reward, done, info = agent.task.step(action)
            state = next_state

            if done:
                break


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def serialize(data):
    if type(data) == list:
        return [serialize(x) for x in data]
    elif type(data) == tuple:
        return tuple([serialize(x) for x in data])
    elif type(data) == dict:
        sdata = {}
        for key, val in data.items():
            sdata['{}'.format(key)] = serialize(val)
        return sdata
    elif type(data) == float or type(data) == int:
        return data
    elif data is None:
        return data
    else:
        if hasattr(data, '__name__'):
            return data.__name__
        else:
            return '{}'.format(data)


def update_seed_in_tag(tag, seed, seed_name, initial_seed=0):
    if seed == initial_seed:
        tag += '-{}={}'.format(seed_name, seed)
    else:
        tag = re.sub(r'-{}=\d+'.format(seed_name), '', tag)
        tag += '-{}={}'.format(seed_name, seed)
    return tag


def dict_config(config):
    config_dict = {}
    for name in filter(lambda s: not re.match('__', s), dir(config)):
        if getattr(config, name) and name not in ['print_every', 'save_every', 'set']:
            config_dict[name] = getattr(config, name)
    return config_dict



