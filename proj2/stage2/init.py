import os
import sys

STAGE = 'stage1'


def is_ipython_mode():
    try:
        _ = __file__
    except:
        return True
    return False


def _init():
    try:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        curr_dir = os.path.abspath(os.path.curdir)

    proj_dir = os.path.dirname(curr_dir)

    sys.path.append(proj_dir)


_init()
