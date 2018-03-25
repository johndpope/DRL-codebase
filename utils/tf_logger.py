# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

from tensorboardX import SummaryWriter
import logging


class Logger(object):
    def __init__(self, log_dir):
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        self.info = logger.info
        self.debug = logger.debug
        self.setLevel = logger.setLevel

        self.writer = SummaryWriter(log_dir)