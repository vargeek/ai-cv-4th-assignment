#!/usr/bin/env python

# %%
import init
from common import generate_train_test_list as base
import iou_pixel
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
    parser, arg = base.get_args_parser('GenerateTrainTestList')

    arg('--negative-ratio', type=float, default=3,
        help='negative sample count over positive sample count')

    arg('--negative-iou', type=float, default=0.3,
        help='min iou of positive sample')

    arg('--show-neg', action='store_true', default=False,
        help='show negative samples')

    args = parser.parse_args(args)
    args.stage = init.STAGE

    return args


class Loader(base.Loader):
    def process_all_lines(self, lines):
        """
        处理每行数据。子类可以重写此方法来实现不同的处理
        param lines: map<(str, list<int>, list<float>, (int, int))>
        """
        lines = list(self.map_expand_roi(lines))
        len_lines = len(lines)

        # 按文件名分组
        groups = {}
        for line in lines:
            filename = line[0]
            group = groups.get(filename)
            if group is None:
                group = []
                groups[filename] = group
            group.append(line)

        samples = []
        len_expected = math.floor(len(lines) * self.args.negative_ratio)

        max_attempt_times = len_expected * 1000
        attempt_times = 0

        while len(samples) < len_expected and attempt_times < max_attempt_times:
            attempt_times = attempt_times + 1
            idx = random.randint(0, len_lines - 1)
            line = lines[idx]
            filename = line[0]
            img_width, img_height = line[3]
            rect = line[1]
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            x = random.randint(0, img_width - w)
            y = random.randint(0, img_height - h)
            roi = [x, y, x+w-1, y+h-1]

            group = groups.get(filename)
            if self.is_negative_sample(roi, group):
                samples.append((filename, roi, []))

        lines.extend(samples)

        return lines

    def is_negative_sample(self, roi, group):
        """
        判断是否为负样本
        param group: list<(str, list<int>, list<float>, (int, int))>
        """
        if group is None:
            return False

        max_iou = self.args.negative_iou

        for line in group:
            roi2 = line[1]
            if iou_pixel.IOU(roi, roi2) > max_iou:
                return False

        return True

    def show_neg_samples(self):
        lines = self.parse_metadata()
        lines = self.process_all_lines(lines)
        lines = filter(lambda line: len(line[2]) == 0, lines)

        for line in lines:
            self.show_image(line)


if __name__ == "__main__":
    seed = 1  # 写死随机种子，方便调试
    random.seed(seed)
    np.random.seed(seed)
    args = _parse_args([] if init.is_ipython_mode() else None)

    print(args)

    loader = Loader(args)

    if args.show_neg:
        loader.show_neg_samples()
    elif args.show:
        image_name = args.image_name
        if image_name is None:
            loader.show_image_demo()
        else:
            loader.show_image_for_name(image_name)
    else:
        loader.generate_train_test_list()


# %%
