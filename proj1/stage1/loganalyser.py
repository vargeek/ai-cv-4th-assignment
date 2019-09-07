#!/usr/bin/env python
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

IPYTHON_MODE = 'get_ipython' in dir()
CUR_DIR = os.path.curdir if IPYTHON_MODE else os.path.dirname(
    __file__)
sys.path.append(os.path.join(CUR_DIR, '..'))


if __name__ == "__main__":
    from utils import LogAnalyser

    LogAnalyser.main([] if IPYTHON_MODE else None)
