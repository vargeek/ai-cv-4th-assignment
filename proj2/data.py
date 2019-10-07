#!/usr/bin/env python
# %%
import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

folder_list = ['I', 'II']
train_boarder = 112
epsilon = 0.0000001


def channel_norm(img):
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + epsilon)
    return pixels


def parse_line(line):
    """
    `str` => [`str`, `int`x4, `float`x42]
    """
    tokens = line.strip().split()
    img_name = tokens[0]
    rect = list(map(lambda x: int(float(x)), tokens[1:5]))
    landmarks = list(map(float, tokens[5:]))
    return img_name, rect, landmarks


class Normalize(object):
    """
    Resieze to train_boarder x train_boarder. Here we use 112 x 112
    Then do channel normalization: (image - mean) / std_variation
    """

    def __call__(self, sample):
        image, landmarks, image_name = sample['image'], sample['landmarks'], sample['image_name']

        # image_resize = np.asarray(image.resize(
        #     (train_boarder, train_boarder), Image.BILINEAR), dtype=np.float32)

        image = channel_norm(image)

        return {
            'image': image,
            'landmarks': landmarks,
            'image_name': image_name,
        }


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    Tensors channel sequence: N x C x H x W
    """

    def __call__(self, sample):
        image, landmarks, image_name = sample['image'], sample['landmarks'], sample['image_name']

        image = np.expand_dims(image, axis=0)

        return {
            'image': torch.from_numpy(image),
            'landmarks': torch.from_numpy(landmarks),
            'image_name': image_name,
        }


class FaceLandmarksDataset(Dataset):
    def __init__(self, data_dir, src_lines, transform=None, cache_in_memory=True):
        """
        @param src_lines: list<str>  
        @param transform: data transform
        """
        self.data_dir = data_dir
        self.lines = src_lines
        self.transform = transform
        if cache_in_memory:
            self.cache = {}
        else:
            self.cache = None

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if self.cache is not None:
            sample = self.cache.get(idx)
            if sample is not None:
                return sample

        img_name, rect, landmarks = parse_line(self.lines[idx])
        img_path = os.path.join(self.data_dir, img_name)

        img = Image.open(img_path).convert('L')
        img_crop = img.crop(tuple(rect))
        w, h = img_crop.size

        img_crop = img_crop.resize(
            (train_boarder, train_boarder), Image.BILINEAR)
        img_crop = np.array(img_crop, dtype=np.float32)

        landmarks = np.array(landmarks, dtype=np.float32)

        # you should let your landmarks fit to the train_boarder(112)
        # please complete your code under this blank
        # your code:
        x0, y0, *_ = rect
        landmarks[0::2] = (landmarks[0::2] - x0) * train_boarder / w
        landmarks[1::2] = (landmarks[1::2] - y0) * train_boarder / h

        sample = {
            'image': img_crop,
            'landmarks': landmarks,
            'image_name': img_name,
        }
        sample = self.transform(sample)
        if self.cache is not None:
            self.cache[idx] = sample
        return sample


def load_data(phase, data_dir=None, cache_in_memory=True):
    if data_dir is None:
        import util
        data_dir = util.get_data_dir()
    filepath = os.path.join(data_dir, phase + '.txt')
    with open(filepath) as f:
        lines = f.readlines()

    if phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor(),
        ])
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor(),
        ])

    data_set = FaceLandmarksDataset(
        data_dir, lines, transform=tsfm, cache_in_memory=cache_in_memory)
    return data_set


def get_train_test_set(data_dir=None, cache_in_memory=True):
    train_set = load_data('train', data_dir, cache_in_memory)
    valid_set = load_data('test', data_dir, cache_in_memory)
    return train_set, valid_set


def _parse_args():
    from util import get_args_parser
    parser, p = get_args_parser('data')

    p('--phase', type=str, metavar='train|test',
          default='train', help='train or test')

    args = parser.parse_args()
    args.phase = args.phase.lower()

    return args



# %%
if __name__ == "__main__":
    args = _parse_args()

    dataset = load_data(args.phase)

    for i in range(0, len(dataset)):
        sample = dataset[i]
        img = sample['image']
        landmarks = sample['landmarks']
        image_name = sample['image_name']
        # 请画出人脸crop以及对应的landmarks
        # please complete your code under this blank

        plt.title(image_name)
        plt.imshow(img[0])
        plt.scatter(landmarks[0::2], landmarks[1::2],
                    alpha=0.5, color='r')
        plt.show()
