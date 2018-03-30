# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

from past.utils import old_div
import numpy as np


# -- UNIFORM


def uniform(low, high, rng=None, size=()):
    return rng.uniform(low, high, size=size)


def loguniform(low, high, rng=None, size=()):
    draw = rng.uniform(low, high, size=size)
    return np.exp(draw)


def quniform(low, high, q, rng=None, size=()):
    draw = rng.uniform(low, high, size=size)
    return np.round(old_div(draw, q)) * q


def qloguniform(low, high, q, rng=None, size=()):
    draw = np.exp(rng.uniform(low, high, size=size))
    return np.round(old_div(draw, q)) * q


# -- NORMAL


def normal(mu, sigma, rng=None, size=()):
    return rng.normal(mu, sigma, size=size)


def qnormal(mu, sigma, q, rng=None, size=()):
    draw = rng.normal(mu, sigma, size=size)
    return np.round(old_div(draw, q)) * q


def lognormal(mu, sigma, rng=None, size=()):
    draw = rng.normal(mu, sigma, size=size)
    return np.exp(draw)


def qlognormal(mu, sigma, q, rng=None, size=()):
    draw = np.exp(rng.normal(mu, sigma, size=size))
    return np.round(old_div(draw, q)) * q