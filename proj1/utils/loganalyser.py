#!/usr/bin/env python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

actions = {
    'train': 'show_train_loss',
}


class LogAnalyser():
    def __init__(self, args, actions=actions):
        self.args = args
        self.actions = actions

    def run(self):
        args = self.args
        action = args.action
        if action is None:
            raise Exception('action is None')

        fn_name = self.actions.get(action.lower())
        if fn_name is None:
            raise Exception('unknown action: {}'.format(action))
        getattr(self, fn_name)()

    @classmethod
    def get_args_parser(self, actions=actions, default_action='train', args=None):
        from utils.util import get_args_parser

        parser, arg = get_args_parser('logloader')
        actions_str = '|'.join(actions.keys())

        arg('--save-directory', type=str, default='out', help='save directory')

        arg('--execid', type=str, default=None, help='sub directory')

        arg('--action', '-a', metavar=actions_str,
            type=str, default=default_action, help='actions')

        parse_args = parser.__getattribute__('parse_args')

        def _parse_args(*args, **kwargs):
            args = parse_args(*args, **kwargs)
            args.root_log_file = 'log.log'
            args.sub_log_file = 'log.log'
            return args

        parser.__setattr__('parse_args', _parse_args)

        return parser, arg

    @staticmethod
    def readlines(logpath, filterfn=None):
        lines = []
        with open(logpath) as f:
            lines = f.readlines()
        retval = map(json.loads, lines)
        if filterfn is not None:
            retval = filter(filterfn, retval)
        return retval

    def load_rootlog(self):
        args = self.args
        log_file = os.path.join(args.save_directory, args.root_log_file)
        return LogAnalyser.readlines(log_file)

    def load_execlog_df(self, phase, tag=None):
        args = self.args

        execid = self.get_execid(phase)
        logfile = os.path.join(args.save_directory, execid, args.sub_log_file)
        filterfn = (lambda x: x.get('tag') == tag) if tag is not None else None
        return pd.DataFrame(LogAnalyser.readlines(logfile, filterfn))

    def get_execid(self, phase):
        args = self.args

        if args.execid is None:
            rows = self.load_rootlog()
            rows = filter(lambda x: x.get('phase') == phase, rows)
            last_row = list(rows)[-1]

            args.execid = last_row['execid']

        return args.execid

    def show_train_loss(self):
        data = self.load_execlog_df('train', 'loss')

        train_losses = data['train_loss']
        valid_losses = data['valid_loss']
        train_acc = data['train_acc']
        valid_acc = data['valid_acc']

        fig = plt.figure(0)
        fig.clear()
        plts = fig.subplots(2, 1)
        plt1, plt2 = plts[0], plts[1]

        # plt1.clear()
        plt1.plot(range(len(train_losses)), train_losses, marker='o')
        plt1.plot(range(len(valid_losses)), valid_losses, marker='o')
        plt1.legend(['train_losses', 'valid_losses'])

        # plt2.clear()
        plt2.plot(range(len(train_acc)), train_acc, marker='o')
        plt2.plot(range(len(valid_acc)), valid_acc, marker='o')
        plt2.legend(['train_acc', 'valid_acc'])

        plt.show()
        # plt.show(block=False)
        # plt.pause(0.000001)

    @staticmethod
    def main(args):
        parser, _ = LogAnalyser.get_args_parser()
        LogAnalyser(parser.parse_args(args)).run()
