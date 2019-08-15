import os


def readlines(filepath):
    lines = []
    with open(filepath) as f:
        lines = f.readlines()

    return map(lambda x: x.strip().split(), lines)


def get_proj_dir():
    """
    获取项目路径
    """
    if '__file__' in locals().keys():
        return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    else:
        return os.path.dirname(os.path.realpath(os.curdir))


def get_data_dir():
    """
    获取数据路径
    """
    return os.path.join(get_proj_dir(), 'data')


def parse_line(line):
    """
    解析一行数据  
    line: list<str>
    """
    img_name = line[0]
    rect = list(map(int, list(map(float, line[1:5]))))
    landmarks = list(map(float, line[5:]))
    return img_name, rect, landmarks
