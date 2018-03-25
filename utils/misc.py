# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

import collections
import pickle

def run_episodes(agent):
    agent.config.logger.info('Start')
    episodes_done = 0
    running_reward = 0
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
            with open('data/%s-%s-online-stats-%s.pkl' % (agent.__class__.__name__, agent.config.tag, agent.task.name), 'wb') as f:
                pickle.dump(dict(stats), f)

        if running_reward > agent.task.success_threshold:
            agent.config.logger.info('Solved! Running reward %.2f, last episode reward %d' % (running_reward, reward))
            break

    agent.save('data/%s-%s-model-%s.pkl' % (agent.__class__.__name__, agent.config.tag, agent.task.name))
    with open('data/%s-%s-all-stats-%s.pkl' % (agent.__class__.__name__, agent.config.tag, agent.task.name), 'wb') as f:
        pickle.dump(dict(stats), f)

