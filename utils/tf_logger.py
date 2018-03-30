# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

from tensorboardX import SummaryWriter
import logging
import os


class Logger(object):
    def __init__(self, log_dir=None):
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        self.info = logger.info
        self.debug = logger.debug
        self.setLevel = logger.setLevel

        if log_dir:
            self.writer = SummaryWriter(log_dir)