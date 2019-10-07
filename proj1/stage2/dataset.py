#!/usr/bin/env python
# %%
import torch
import os
import pandas as pd
from Species_Network import Net
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

IPYTHON_MODE = 'get_ipython' in dir()
CUR_DIR = os.path.curdir if IPYTHON_MODE else os.path.dirname(
    __file__)

train_transform = transforms.Compose([
    # transforms.Resize((500, 500)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
valid_transform = transforms.Compose([
    # transforms.Resize((500, 500)),
    transforms.ToTensor()
])
TRAIN_ANNO = 'Species_train_annotation.csv'
VALID_ANNO = 'Species_val_annotation.csv'
phases = {
    'train': (TRAIN_ANNO, train_transform),
    'valid': (VALID_ANNO, valid_transform),
}
SPECIES = ['rabbits', 'rats', 'chickens']


class SpeciesDataset(Dataset):
    def __init__(self, root_dir, annotations, transform=None, cache_in_memory=True, path_field=False):
        self.root_dir = root_dir
        self.resize_transform = transforms.Resize((500, 500))
        self.transform = transform
        self.annotations = annotations
        self.cached = {} if cache_in_memory else None
        self.path_field = path_field

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations.iloc[idx]

        path = os.path.join(self.root_dir, item['path'])

        image = self.cached.get(idx) if self.cached is not None else None
        if image is None:
            image = Image.open(path)
            if image.palette:
                image = image.convert('RGBA').convert('RGB')
                # image = image.convert('RGB')
            else:
                image = image.convert('RGB')

            image = self.resize_transform(image)

            if self.cached is not None:
                self.cached[idx] = image

        if self.transform:
            image = self.transform(image)

        species = int(item['species'])

        sample = {
            'image': image,
            'species': species,
        }
        if self.path_field:
            sample['path'] = item['path']

        return sample


def load_annotations_file(filename):
    if not os.path.isabs(filename):
        filename = os.path.join(CUR_DIR, filename)

    return pd.read_csv(filename, index_col=0)


def load_dataset(args, phase, path_field):
    filename, transform = phases[phase]

    data = load_annotations_file(filename)
    return SpeciesDataset(args.data_directory, data, transform, not args.no_cache_image, path_field)


def get_train_test_set(args, path_field=False):
    train_set = load_dataset(args, 'train', path_field)
    valid_set = load_dataset(args, 'valid', path_field)

    return train_set, valid_set


def _get_args(args=None):
    import sys
    import os
    sys.path.append(os.path.join(CUR_DIR, '..'))
    from utils import util

    parser, arg = util.get_args_parser('dataset')

    arg('--data-directory', type=str,
        help='data are loading from here')

    arg('--count', type=int, default=10,
        help='samples count to show')

    arg('--phase', type=str, default='train',
        metavar='train|valid',
        help='dataset of which phase to show')

    arg('--path', '-p', type=str, nargs='+', default=None,
        help='path of image to show')

    return parser.parse_args(args)


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    args = _get_args([] if IPYTHON_MODE else None)
    args.no_cache_image = False

    train_set, val_set = get_train_test_set(args, True)
    print('length of train_set: ', len(train_set))
    print('length of valid_set: ', len(val_set))

    is_train_set = args.phase == 'train'
    dataset = train_set if is_train_set else val_set

    count = args.count

    if args.path is not None:
        prefix = 'train/' if is_train_set else 'val/'
        paths = [prefix + p for p in args.path]
        dataset = [d for d in dataset if d['path'] in paths]
        count = len(dataset)

    for _ in range(count):
        idx = random.randint(0, len(dataset)-1)
        sample = dataset[idx]
        print(sample['path'])
        print(idx, sample['image'].shape, SPECIES[sample['species']])

        img = sample['image']
        plt.imshow(transforms.ToPILImage()(img))
        plt.title('{}: {}'.format(idx, SPECIES[sample['species']]))
        plt.show()


# %%
