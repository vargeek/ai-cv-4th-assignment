#!/usr/bin/env python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt


_parser = None

actions = {
    'train': 'show_train_loss',
}


def _get_args_parser():
    global _parser
    if _parser is None:
        import util
        _parser, p = util.get_args_parser('logloader')

        p('--save-directory', type=str, default='out', help='save directory')

        p('--uuid', type=str, default=None, help='sub directory')

        actions_str = '|'.join(actions.keys())
        p('--action', '-a', metavar=actions_str,
            type=str, default=None, help='sub directory')

    return _parser


def _get_args():
    import util
    args = _get_args_parser().parse_args()
    util.pre_process_args(args)
    args.root_log_file = 'log.log'
    args.sub_log_file = 'log.log'

    return args


def readlines(logpath, filterfn=None):
    lines = []
    with open(logpath) as f:
        lines = f.readlines()
    retval = map(json.loads, lines)
    if filterfn is not None:
        retval = filter(filterfn, retval)
    return retval


class LogLoader():
    def __init__(self, args=_get_args_parser().parse_args([])):
        self.args = args

    def load_rootlog(self):
        args = self.args
        log_file = os.path.join(args.save_directory, args.root_log_file)
        return readlines(log_file)

    def load_sublog_df(self, phase, tag=None):
        args = self.args

        uuid = self.get_uuid(phase)
        logfile = os.path.join(args.save_directory, uuid, args.sub_log_file)
        filterfn = (lambda x: x.get('tag') == tag) if tag is not None else None
        return pd.DataFrame(readlines(logfile, filterfn))

    def show_train_loss(self):
        data = self.load_sublog_df('train', 'loss')
        steps = range(len(data))
        plt.plot(steps, data['train_loss'])
        plt.plot(steps, data['valid_loss'])
        plt.legend(['train_losses', 'valid_losses'])
        plt.show()

    def get_uuid(self, phase):
        args = self.args

        if args.uuid is None:
            rows = self.load_rootlog()
            rows = filter(lambda x: x.get('phase') == phase, rows)
            last_row = list(rows)[-1]

            args.uuid = last_row['uuid']

        return args.uuid


if __name__ == "__main__":
    args = _get_args()
    action = 'train' if args.action is None else args.action
    action = action.lower()
    fn_name = actions.get(action)
    if fn_name is None:
        raise Exception('unknown action: {}'.format(action))

    loader = LogLoader(args)
    getattr(loader, fn_name)()
