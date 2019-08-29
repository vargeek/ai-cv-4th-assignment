# %%
import logging
import json
import os
import sys
import time


class Encoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        if hasattr(o, '__dict__'):
            return o.__dict__
        return super(Encoder, self).default(o)


def file_logger(name, filepath, format='%(message)s'):
    assert(name)
    assert(filepath)

    logger = logging.getLogger(name)
    dirpath = os.path.realpath(os.path.dirname(filepath))
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    if len(logger.handlers) > 0:
        return logger

    hd = logging.FileHandler(filepath)
    formatter = logging.Formatter(format)
    hd.setFormatter(formatter)
    logger.addHandler(hd)
    logger.setLevel(logging.INFO)
    return logger


def std_logger(name, format='[%(asctime)s] <%(funcName)s@%(filename)s:%(lineno)d> \n%(message)s'):
    assert(name)

    logger = logging.getLogger(name)
    if len(logger.handlers) > 0:
        return logger

    hd = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(format)
    hd.setFormatter(formatter)
    logger.addHandler(hd)
    logger.setLevel(logging.INFO)
    return logger


def curr_time_str():
    ct = time.time()
    ms = int((ct - int(ct)) * 1000)
    return '{},{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), ms)


def get_struct_log(log, **kwargs):
    def _log(**_kwargs):
        _kwargs.update(kwargs)
        log(json.dumps(_kwargs, cls=Encoder))
    return _log
