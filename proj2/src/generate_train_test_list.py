# %%
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import util
from PIL import Image
import os

# %%
if not '__file__' in locals().keys():
    if os.path.basename(os.curdir) != 'src':
        if os.path.isdir('proj2'):
            os.chdir('proj2')
        if os.path.isdir('src'):
            os.chdir('src')
    print('curdir:', os.path.realpath(os.curdir))


# %%
proj_dir = util.get_proj_dir()
data_dir = util.get_data_dir()
data_subfolders = ['I', 'II']
metadata_filename = 'label.txt'
train_filename = 'train.txt'
test_filename = 'test.txt'
format = """
____________________________________
proj_dir: {}
data_dir: {}
data_subfolders: {}
metadata_filename: {}
train_filename: {}
test_filename: {}
____________________________________
"""
print(format.format(
    proj_dir,
    data_dir,
    data_subfolders,
    metadata_filename,
    train_filename,
    test_filename))

# %%


def load_metadata(data_subfolders):
    """
    加载 `data/folder/label.txt`
    retval: list<list<str>>
    """
    all_lines = []

    def isFile(line):
        return os.path.isfile(os.path.join(data_dir, line[0]))

    for folder in data_subfolders:
        def prepend_folder(line):
            line[0] = os.path.join(folder, line[0])
            return line

        filepath = os.path.join(data_dir, folder, metadata_filename)
        lines = util.readlines(filepath)

        lines = filter(isFile, map(prepend_folder, lines))

        all_lines.extend(lines)
    return all_lines


# %%


def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio):
    """
    扩增兴趣区域
    param ratio: 四周填充的`padding`比例
    """
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    padding_width = int(width * ratio)
    padding_height = int(height * ratio)

    roi_x1 = x1 - padding_width
    roi_y1 = y1 - padding_height
    roi_x2 = x2 + padding_width
    roi_y2 = y2 + padding_height

    roi_x1 = roi_x1 if roi_x1 >= 0 else 0
    roi_y1 = roi_y1 if roi_y1 >= 0 else 0
    roi_x2 = roi_x2 if roi_x2 < img_width else img_width - 1
    roi_y2 = roi_y2 if roi_y2 < img_height else img_height - 1

    return roi_x1, roi_y1, roi_x2, roi_y2, roi_x2 - roi_x1 + 1, roi_y2 - roi_y1 + 1


def expand_roi_for_lines(lines, ratio):
    """
    扩增多行数据兴趣区域
    param lines: list<list<str>>
    """
    for line in lines:
        img_path = os.path.join(data_dir, line[0])
        rect = map(lambda x: int(float(x)), line[1:5])

        # img = Image.open(img_path).convert('L')
        img = Image.open(img_path)
        img_size = img.size

        roi_x1, roi_y1, roi_x2, roi_y2, * \
            _ = expand_roi(*rect, *img_size, ratio)

        line[1:5] = map(str, [roi_x1, roi_y1, roi_x2, roi_y2])

    return lines
# %%


def generate_train_test_list(data_subfolders=data_subfolders, ratio=0.25, train_ratio=0.7):
    """
    生成训练样本集和测试样本集
    """
    all_lines = expand_roi_for_lines(load_metadata(data_subfolders), ratio)

    all_lines = list(map(' '.join, all_lines))

    num_samples = len(all_lines)
    num_train = int(num_samples * train_ratio)
    np.random.shuffle(all_lines)

    with open(os.path.join(data_dir, train_filename), 'w') as f:
        f.write('\n'.join(all_lines[0:num_train]))

    with open(os.path.join(data_dir, test_filename), 'w') as f:
        f.write('\n'.join(all_lines[num_train:]))

# %%


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


def show_image(line):
    """
    显示图像
    """
    img_name, rect, landmarks = line

    img_path = os.path.join(data_dir, img_name)
    img = Image.open(img_path)
    npimg = np.array(img)
    npimg = np.transpose(npimg, (2, 0, 1))

    _, ax = plt.subplots()
    ax.imshow(img)

    draw_rect(ax, rect)

    ax.scatter(landmarks[0::2], landmarks[1::2], alpha=0.5, c='r')


def show_images(lines1, lines2):
    """
    显示图像
    """
    assert(len(lines1) == len(lines2))
    nrows = len(lines1)
    ncols = 2
    imgs = [lines1, lines2]

    _, axs = plt.subplots(nrows, ncols)

    for col in range(0, ncols):
        for row in range(0, nrows):
            ax = axs[row][col]
            img_name, rect, landmarks = imgs[col][row]

            img_path = os.path.join(data_dir, img_name)
            img = Image.open(img_path)
            npimg = np.array(img)
            npimg = np.transpose(npimg, (2, 0, 1))

            ax.imshow(img)

            draw_rect(ax, rect)

            ax.scatter(landmarks[0::2], landmarks[1::2],
                       alpha=0.5, color='r', s=1)


def show_image_demo(ratio=0.25):
    ori_lines = load_metadata(['I'])
    roi_lines = expand_roi_for_lines(load_metadata(['I']), ratio)

    ori_lines = list(map(util.parse_line, ori_lines))
    roi_lines = list(map(util.parse_line, roi_lines))

    nrows = 10
    # show_images(ori_lines[0:nrows], roi_lines[0:nrows])

    for i in range(nrows):
        show_image(ori_lines[i])
        show_image(roi_lines[i])

    plt.show()


# %%
if __name__ == "__main__":
    # generate_train_test_list(['I'])
    show_image_demo()


# %%
