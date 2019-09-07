#!/usr/bin/env python
# %%
import os
from PIL import Image
import pandas as pd

IPYTHON_MODE = 'get_ipython' in dir()
CUR_DIR = os.path.curdir if IPYTHON_MODE else os.path.dirname(
    __file__)


PHASE = ['train', 'val']
CLASSES = ['Mammals', 'Birds']  # [0,1]
SPECIES = ['rabbits', 'chickens']


def _get_filename(phase):
    return 'Classes_{}_annotation.csv'.format(phase)


def _get_args(args=None):
    import sys
    import os
    sys.path.append(os.path.join(CUR_DIR, '..'))
    from utils import util

    parser, arg = util.get_args_parser('Classes_make_anno')

    arg('--data-directory', type=str,
        help='data are loading from here'),

    return parser.parse_args(args)


def make_anno(args):
    for phase in PHASE:
        items = []
        for i, s in enumerate(SPECIES):
            dirname = os.path.join(phase, s)
            dirpath = os.path.join(args.data_directory, dirname)
            files = os.listdir(dirpath)

            def item(name):
                fullpath = os.path.join(dirpath, name)
                try:
                    Image.open(fullpath)
                except:
                    return None

                return {
                    'path': os.path.join(dirname, name),
                    'classes': i,
                }

            _items = filter(lambda x: x is not None, map(item, files))

            items.extend(_items)

        df = pd.DataFrame(items)
        csv_path = _get_filename(phase)
        df.to_csv(csv_path)
        print('{} file is saved.'.format(csv_path))


if __name__ == "__main__":
    args = _get_args([] if IPYTHON_MODE else None)
    make_anno(args)


# %%
