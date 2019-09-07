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
    try:
        return os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    except:
        return os.path.realpath(os.curdir)


def get_data_dir():
    """
    获取数据路径
    """
    return os.path.join(get_proj_dir(), 'Dataset')


def parse_line(line):
    """
    解析一行数据
    line: list<str>
    """
    img_name = line[0]
    rect = list(map(int, list(map(float, line[1:5]))))
    landmarks = list(map(float, line[5:]))
    return img_name, rect, landmarks


def stringnify_sample_line(line):
    """
    param line: (str, list<int>, list<float>, ...)
    retval: str
    """

    return " ".join([line[0], *map(str, line[1]), *map(str, line[2])])


def _pre_process_args(args):
    if hasattr(args, 'data_directory') and args.data_directory is None:
        args.data_directory = get_data_dir()
    return args


def p(*args, **kwargs):
    return args, kwargs


def get_args_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)

    parse_args = parser.__getattribute__('parse_args')

    def _parse_args(*args, **kwargs):
        args = parse_args(*args, **kwargs)
        return _pre_process_args(args)

    parser.__setattr__('parse_args', _parse_args)

    return parser, parser.add_argument
