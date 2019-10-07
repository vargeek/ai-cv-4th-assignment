import os
import matplotlib.pyplot as plt


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
        return os.path.dirname(os.path.realpath(__file__))
    except:
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
    return args


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

    parse_args = parser.__getattribute__('parse_args')

    def _parse_args(*args, **kwargs):
        args = parse_args(*args, **kwargs)
        return pre_process_args(args)

    parser.__setattr__('parse_args', _parse_args)

    return parser, parser.add_argument

def draw_losses(losses, block=True, min_loss_key='valid_loss'):

    fig = plt.figure(0)
    fig.clear()

    plt1 = fig.subplots(1, 1)
    # plt1, plt2 = plts[0], plts[1]

    title = ''
    if min_loss_key:
        loss = losses.get(min_loss_key)
        if loss is not None:
            min_loss = min(loss)
            title = '{}: {:.6f}'.format(min_loss_key, min_loss)
    # for acc_key in max_acc_keys:
    #     acc = accs.get(acc_key)
    #     if acc is not None:
    #         max_acc = max(acc)
    #         title = '{}, {}: {:.6f}'.format(title, acc_key, max_acc)

    plt1.set_title(title)

    # plt1.clear()
    for v in losses.values():
        plt1.plot(range(len(v)), v, marker='.')
    plt1.legend(losses.keys())

    # plt2.clear()
    # for v in accs.values():
    #     plt2.plot(range(len(v)), v, marker='.')
    # plt2.legend(accs.keys())

    plt.show(block=block)
    if not block:
        plt.pause(0.000001)
