# -*- coding: utf-8 -*-

import logging
import os
import sys


def set_logging(rank=0, dir_root='run', name=''):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter(f"{name}: %(asctime)s-%(message)s")
        handler_file = logging.FileHandler(os.path.join(dir_root, f"{name}.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
