import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys
from . import util


def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio):
    """
    扩增兴趣区域
    param ratio: 四周填充的`padding`比例
    """
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    padding_width = int(width * ratio)
    padding_height = int(height * ratio)

    roi_x1 = max(x1 - padding_width, 0)
    roi_y1 = max(y1 - padding_height, 0)
    roi_x2 = min(x2 + padding_width, img_width - 1)
    roi_y2 = min(y2 + padding_height, img_height - 1)

    return roi_x1, roi_y1, roi_x2, roi_y2, roi_x2 - roi_x1 + 1, roi_y2 - roi_y1 + 1


def draw_rect(ax, rect, color='deepskyblue'):
    x1, y1, x2, y2 = rect
    verts = [
        (x1, y1),  # left, bottom
        (x1, y2),  # left, top
        (x2, y2),  # right, top
        (x2, y1),  # right, bottom
        (0., 0.),  # ignored
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=2, edgecolor=color)
    ax.add_patch(patch)


def get_args_parser(description):
    parser, arg = util.get_args_parser(description)

    arg('--ratio', type=float, default=0.9,
        help='训练集所占比例(default: 0.9)')

    arg('--expand-ratio', type=float, default=0.25,
        help='roi扩增比例(default: 0.25)')

    arg('--data-directory', type=str,
        help='图片路径')

    arg('--folders', type=str, nargs='+', default=['I', 'II'],
        help='文件夹')

    arg('--show', action='store_true', default=False,
        help='show images')

    arg('--image-name', type=str,
        help='图片名称')

    arg('--stage', type=str, default='stage1',
        help='data are loading from here')

    parse_args = parser.__getattribute__('parse_args')

    def _parse_args(*args, **kwargs):
        args = parse_args(*args, **kwargs)
        args.metadata_filename = 'label.txt'
        args.train_filename = 'train.txt'
        args.test_filename = 'test.txt'
        return args

    parser.__setattr__('parse_args', _parse_args)
    parser.parse_args

    return parser, arg


def _parse_args(args=None):
    parser, _ = get_args_parser('GenerateTrainTestList')

    return parser.parse_args(args)


class Loader():
    def __init__(self, args):
        self.args = args

    def load_metadata(self):
        """
        加载 `data/folder/label.txt`
        retval: list<list<str>>
        """
        data_dir = self.args.data_directory
        metadata_filename = self.args.metadata_filename

        all_lines = []

        def isFile(line):
            return os.path.isfile(os.path.join(data_dir, line[0]))

        for folder in self.args.folders:
            def prepend_folder(line):
                line[0] = os.path.join(folder, line[0])
                return line

            filepath = os.path.join(
                data_dir, folder, metadata_filename)

            lines = util.readlines(filepath)
            lines = filter(isFile, map(
                prepend_folder, lines))

            all_lines.extend(lines)

        return all_lines

    def parse_metadata(self):
        """
        加载并解析 `data/folder/label.txt`
        retval: (str, list<int>, list<float>, (int, int))

        """
        lines = self.load_metadata()
        lines = self.parse_lines(lines)
        lines = self.filter_outliers(lines)
        return list(lines)

    def parse_lines(self, lines):
        """
        param lines: map<list<str>>
        retval: (str, list<int>, list<float>, (int, int))
        """

        def mapfn(line):
            line = util.parse_line(line)
            img_path = os.path.join(self.args.data_directory, line[0])
            img = Image.open(img_path)
            img_size = img.size
            return (*line, img_size)

        return map(mapfn, lines)

    def stringnify_lines(self, lines):
        """
        param lines: (str, list<int>, list<float>, (int, int))
        retval: map<str>
        """
        def mapfn(line):
            return " ".join([line[0], *map(str, line[1]), *map(str, line[2])])
        return map(mapfn, lines)

    def filter_outliers(self, lines):
        """
        param lines: filter<(str, list<int>, list<float>, (int, int))>
        """
        def filterfn(line):
            w, h = line[3]

            return all(x >= 0 and x < w for x in line[2][0::2]) and all(y >= 0 and y < h for y in line[2][1::2])

        return filter(filterfn, lines)

    def map_expand_roi(self, lines):
        """
        param lines: map<(str, list<int>, list<float>, (int, int))>
        """
        ratio = self.args.expand_ratio

        def mapfn(line):
            roi_x1, roi_y1, roi_x2, roi_y2, * \
                _ = expand_roi(*line[1], *line[3], ratio)

            return (line[0], [roi_x1, roi_y1, roi_x2, roi_y2], *line[2:])

        return map(mapfn, lines)

    def process_all_lines(self, lines):
        """
        处理每行数据。子类可以重写此方法来实现不同的处理  
        param lines: map<(str, list<int>, list<float>, (int, int))>
        """
        return self.map_expand_roi(lines)

    def generate_train_test_list(self):
        """
        生成训练样本集和测试样本集
        """
        args = self.args
        ratio = args.ratio
        data_dir = args.data_directory
        sample_dir = os.path.join(data_dir, args.stage)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        all_lines = self.parse_metadata()
        all_lines = self.process_all_lines(all_lines)

        all_lines = list(self.stringnify_lines(all_lines))

        num_samples = len(all_lines)
        num_train = int(num_samples * ratio)
        np.random.shuffle(all_lines)
        with open(os.path.join(data_dir, args.stage, args.train_filename), 'w') as f:
            f.write('\n'.join(all_lines[0:num_train]))

        with open(os.path.join(data_dir, args.stage, args.test_filename), 'w') as f:
            f.write('\n'.join(all_lines[num_train:]))

    def show_image(self, line):
        """
        显示图像
        """
        img_name, rect, landmarks, *_ = line

        img_path = os.path.join(self.args.data_directory, img_name)
        img = Image.open(img_path)
        npimg = np.array(img)
        npimg = np.transpose(npimg, (2, 0, 1))

        _, ax = plt.subplots()
        ax.set_title(img_name)
        ax.imshow(img)

        draw_rect(ax, rect)

        ax.scatter(landmarks[0::2], landmarks[1::2], alpha=0.5, c='r')

        plt.show()

    def show_image_demo(self):

        ori_lines = self.parse_metadata()
        roi_lines = list(self.map_expand_roi(ori_lines))

        for i in range(len(ori_lines)):
            self.show_image(ori_lines[i])
            self.show_image(roi_lines[i])

    def show_image_for_name(self, image_name):

        ori_lines = self.parse_metadata()
        roi_lines = list(self.map_expand_roi(ori_lines))

        for i in range(len(ori_lines)):
            if ori_lines[i][0] == image_name:
                self.show_image(ori_lines[i])
                self.show_image(roi_lines[i])
                break


# %%
if __name__ == "__main__":
    np.random.seed(1)  # 写死随机种子，方便调试

    args = _parse_args()
    print(args)

    loader = Loader(args)

    if args.show:
        image_name = args.image_name
        if image_name is None:
            loader.show_image_demo()
        else:
            loader.show_image_for_name(image_name)
    else:
        loader.generate_train_test_list()
