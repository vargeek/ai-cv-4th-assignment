import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

import util
import os
import math


def gamma_correction(args, img, rect, pts):
    """
    gamma校正
    """
    invGamma = 1.0 / args.gamma
    table = [(x / 255.0)**invGamma * 255 for x in range(256)]
    table = np.array(table).astype("uint8")

    return cv2.LUT(img, table), rect, pts


def rotate_roi(mat, x0, y0, x1, y1):

    pts = [
        mat @ [x0, y0, 1],
        mat @ [x1, y0, 1],
        mat @ [x1, y1, 1],
        mat @ [x0, y1, 1],
    ]
    minx = min(pt[0] for pt in pts)
    maxx = max(pt[0] for pt in pts)
    miny = min(pt[1] for pt in pts)
    maxy = max(pt[1] for pt in pts)

    return minx, miny, maxx, maxy


def int_rect(x0, y0, x1, y1):
    from math import ceil, floor
    return ceil(x0), ceil(y0), floor(x1), floor(y1)


def rotate(args, img, rect, pts):

    rows, cols, _ = img.shape

    ox = (rect[0] + rect[2]) * 0.5
    oy = (rect[1] + rect[3]) * 0.5

    angle = args.angle
    angle = random.random() * angle * 2 - angle

    matrix = cv2.getRotationMatrix2D((ox, oy), angle, 1)

    x0, y0, x1, y1 = rotate_roi(matrix, 0, 0, cols, rows)

    translate = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1],
    ])

    matrix = translate @ np.vstack((matrix, [0, 0, 1]))
    matrix = matrix[0:2, :]

    new_cols = math.ceil(abs(x1 - x0))
    new_rows = math.ceil(abs(y1 - y0))

    rotated_img = cv2.warpAffine(img, matrix, (new_cols, new_rows))
    rotated_rect = int_rect(*rotate_roi(matrix, *rect))

    rotated_pts = map(lambda x: matrix @ [
                      x[0], x[1], 1], zip(pts[0::2], pts[1::2]))
    rotated_pts = np.array(list(rotated_pts)).flatten()

    return rotated_img, rotated_rect, rotated_pts


def flipx(args, img, rect, pts):
    _, cols, *_ = img.shape
    img = img[:, ::-1, :]

    x0, y0, x1, y1 = rect
    rect = [cols - x1-1, y0, cols - x0-1, y1]

    xs = [cols - x - 1 for x in pts[0::2]]
    ys = pts[1::2]
    pts = np.array(list(zip(xs, ys))).flatten()

    return img, rect, pts


def augmentation_line(args, tokens):
    from_dir = args.from_directory
    save_dir = args.save_directory

    img_name, rect, pts = util.parse_line(tokens)

    img = cv2.imread(os.path.join(from_dir, img_name))
    rows, cols, *_ = img.shape

    is_valid = all(x >= 0 and x < cols for x in pts[0::2]) and all(
        y >= 0 and y < rows for y in pts[1::2])
    if not is_valid:
        return None

    for fn in args.augmentation:
        img, rect, pts = fn(args, img, rect, pts)

    cv2.imwrite(os.path.join(save_dir, img_name), img)

    line_str = util.stringnify_sample_line([
        img_name,
        rect,
        pts,
    ])
    return line_str


def generate_samples(args):

    data_dir = args.data_directory
    for dir_tuple in args.dirs:
        args.save_directory = os.path.join(data_dir, dir_tuple[1])
        args.from_directory = os.path.join(data_dir, dir_tuple[0])
        args.from_file = os.path.join(args.from_directory, args.filename)
        args.save_file = os.path.join(args.save_directory, args.filename)

        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

        lines = iter(augmentation_line(args, tokens)
                     for tokens in util.readlines(args.from_file)
                     )
        lines = list(filter(lambda x: x is not None, lines))

        # lines = [augmentation_line(args, tokens)
        #          for tokens in [next(util.readlines(args.from_file))]]

        text = '\n'.join(lines)

        with open(args.save_file, 'w') as f:
            f.write(text)


def _parse_args():
    from util import p
    args = util.parse_args('augmentation', [
        p('--data-directory', type=str, default=None, help='data directory'),

        p('--dirs', '-d', type=str, nargs='+',
          default=['I:III', 'II:IV'], help='sub directories'),

        p('--augmentation', '-a', type=str, nargs='+',
          default=['flipx'],
          metavar='rotate|flipx|gamma_correction',
          help='augmentation actions'),

        p('--gamma', type=float,
          default=2.0, help='gamma'),

        p('--angle', type=float, default='10',
          help='max rotate angle'),
    ])

    args.dirs = [[x.strip() for x in d.strip().split(':')] for d in args.dirs]
    args.filename = 'label.txt'

    return args


if __name__ == "__main__":
    args = _parse_args()
    fns = locals()

    args.augmentation = list(
        filter(lambda x: x is not None, map(fns.get, args.augmentation)))

    print(args)

    generate_samples(args)
