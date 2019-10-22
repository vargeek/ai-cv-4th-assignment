#!/usr/bin/env python

# %%
import init
from common import generate_train_test_list as base
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from PIL import Image
import os
import sys


# %%


def _parse_args(args=None):
    import common
    parser, _ = base.get_args_parser('GenerateTrainTestList')

    args = parser.parse_args(args)
    args.stage = init.STAGE

    return args


if __name__ == "__main__":
    seed = 1  # 写死随机种子，方便调试
    random.seed(seed)
    np.random.seed(seed)
    args = _parse_args([] if init.is_ipython_mode() else None)

    print(args)

    loader = base.Loader(args)

    if args.show:
        image_name = args.image_name
        if image_name is None:
            loader.show_image_demo()
        else:
            loader.show_image_for_name(image_name)
    else:
        loader.generate_train_test_list()


# %%
