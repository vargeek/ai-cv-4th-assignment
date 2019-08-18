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
        image, landmarks = sample['image'], sample['landmarks']

        image_resize = np.asarray(image.resize(
            (train_boarder, train_boarder), Image.BILINEAR), dtype=np.float32)

        image = channel_norm(image_resize)

        return {
            'image': image,
            'landmarks': landmarks,
        }


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    Tensors channel sequence: N x C x H x W
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        image = np.expand_dims(image, axis=0)

        return {
            'image': torch.from_numpy(image),
            'landmarks': torch.from_numpy(landmarks),
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

        landmarks = np.array(landmarks, dtype=np.float32)

        # you should let your landmarks fit to the train_boarder(112)
        # please complete your code under this blank
        # your code:
        w, h = img_crop.size
        x0, y0, *_ = rect
        landmarks[0::2] = (landmarks[0::2] - x0) * train_boarder / w
        landmarks[1::2] = (landmarks[1::2] - y0) * train_boarder / h

        sample = {
            'image': img_crop,
            'landmarks': landmarks,
        }
        sample = self.transform(sample)
        if self.cache is not None:
            self.cache[idx] = sample
        return sample


def load_data(phase, data_dir=None):
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

    data_set = FaceLandmarksDataset(data_dir, lines, transform=tsfm)
    return data_set


def get_train_test_set(data_dir=None):
    train_set = load_data('train', data_dir)
    valid_set = load_data('test', data_dir)
    return train_set, valid_set


def _get_argparser():
    import argparse
    parser = argparse.ArgumentParser(description='data')

    parser.add_argument('--phase', type=str,
                        default='train', help='train or test')

    return parser


def _parse_args():
    args = _get_argparser().parse_args()
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
        # 请画出人脸crop以及对应的landmarks
        # please complete your code under this blank

        plt.imshow(img[0])
        plt.scatter(landmarks[0::2], landmarks[1::2],
                    alpha=0.5, color='r')
        plt.show()
