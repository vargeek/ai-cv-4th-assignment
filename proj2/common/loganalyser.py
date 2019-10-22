#!/usr/bin/env python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from . import util


class LogAnalyser():
    def __init__(self, args):
        self.args = args
        from .util import show_train_loss
        self._show_train_loss = None

    def get_best_for_rows(self, rows, fn):
        def reduce(fn, rows):
            if len(rows) == 0:
                return None
            best_row = rows.iloc[0]
            for i in range(1, len(rows)):
                best_row = fn(best_row, rows.iloc[i])
            return best_row
        return reduce(fn, rows)

    def get_best_for_columns(self, data, columns):

        result = {}
        for col, fn in columns.items():
            rows = data[['epoch', col]].rename(columns={col: 'val'})
            best_row = self.get_best_for_rows(rows, fn)
            result[col] = best_row

        keys = result.keys()
        values = [result[k] for k in keys]
        return pd.DataFrame(values, index=keys)

    @staticmethod
    def min(row1, row2):
        return row1 if row1['val'] <= row2['val'] else row2

    @staticmethod
    def max(row1, row2):
        return row1 if row1['val'] >= row2['val'] else row2

    def run(self):
        util.get_action_fn(self, self.args.action)()

    @classmethod
    def get_args_parser(self, args=None):
        from .util import get_args_parser, get_actions

        parser, arg = get_args_parser('logloader')
        actions_str = '|'.join(get_actions(self))

        arg('--save-directory', type=str, default='out', help='save directory')

        arg('--execid', type=str, default=None, help='sub directory')

        arg('--action', '-a', metavar=actions_str,
            type=str, default='show', help='actions')

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

    def _show_action_(self):
        data = self.load_execlog_df('train', 'loss')

        fields = ['train_loss', 'valid_loss']
        util.show_metrics(
            data, fields, title_format=util.title_format_for_fields(fields))

    def get_reduced(self, fields):
        data = self.load_execlog_df('train', 'loss')
        import functools

        if len(data) == 0:
            return None

        results = {}
        for name, fn in fields:
            result = functools.reduce(fn, data[name])
            results[name] = {'epoch': result['epoch'], 'val': result[name]}

        return pd.DataFrame(results)

    def _best_action_(self):
        data = self.load_execlog_df('train', 'loss')

        result = self.get_best_for_columns(data, {
            'valid_loss': LogAnalyser.min,
        })

        print(result)

    @classmethod
    def main(clazz, args):
        parser, _ = clazz.get_args_parser()
        clazz(parser.parse_args(args)).run()
