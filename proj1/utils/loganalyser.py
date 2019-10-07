#!/usr/bin/env python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

actions = {
    'train': 'show_train_loss',
    'best': 'get_best_val',
}


class LogAnalyser():
    def __init__(self, args, actions=actions):
        self.args = args
        self.actions = actions
        from utils.util import show_train_loss
        self._show_train_loss = show_train_loss

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

        self._show_train_loss(train_losses, valid_losses, train_acc, valid_acc)

    def get_best_val(self):
        data = self.load_execlog_df('train', 'loss')

        best_loss_epoch = data.iloc[0]['epoch']
        best_acc_epoch = data.iloc[0]['epoch']
        best_loss = data.iloc[0]['valid_loss']
        best_acc = data.iloc[0]['valid_acc']
        for i in range(len(data)):
            item = data.iloc[i]
            epoch = item['epoch']
            valid_loss = item['valid_loss']
            valid_acc = item['valid_acc']

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_loss_epoch = epoch

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_acc_epoch = epoch

        print('best valid loss: epoch: {}, loss: {}'.format(
            best_loss_epoch, best_loss))
        print('best valid acc: epoch: {}, acc: {}'.format(
            best_acc_epoch, best_acc))

    @staticmethod
    def main(args):
        parser, _ = LogAnalyser.get_args_parser()
        LogAnalyser(parser.parse_args(args)).run()
