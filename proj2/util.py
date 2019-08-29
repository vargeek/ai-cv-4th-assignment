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
        return os.path.dirname(os.path.realpath(__file__))
    else:
        return os.path.realpath(os.curdir)


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


def stringnify_sample_line(line):
    """
    param line: (str, list<int>, list<float>, ...)
    retval: str
    """

    return " ".join([line[0], *map(str, line[1]), *map(str, line[2])])


def pre_process_args(args):
    if hasattr(args, 'data_directory') and args.data_directory is None:
        args.data_directory = get_data_dir()


def p(*args, **kwargs):
    return args, kwargs


def parse_args(description, params):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    for param in params:
        args, kwargs = param
        parser.add_argument(*args, **kwargs)

    args = parser.parse_args()
    pre_process_args(args)

    return args


def get_args_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    return parser, parser.add_argument
