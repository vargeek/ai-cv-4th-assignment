#!/usr/bin/env python
import init
from common import loganalyser as base
import pandas as pd
import numpy as np


class LogAnalyser(base.LogAnalyser):

    def _best_action_(self):

        data = self.load_execlog_df('train', 'loss')

        result = self.get_best_for_columns(data, {
            'valid_loss': LogAnalyser.min,
            'val_acc': LogAnalyser.max,
        })

        print(result)

    def _show_action_(self):
        data = self.load_execlog_df('train', 'loss')

        from common import util
        from common.util import title_format_for_fields

        format1 = title_format_for_fields(['valid_loss'])
        format2 = title_format_for_fields(['val_acc'], reduce=max)
        format3 = title_format_for_fields(['pred_val_loss'])
        format4 = title_format_for_fields(['cls_val_loss'])
        formats = [[format1, format2], [format3, format4]]

        def title_format(metrics, row, col):
            format = formats[row][col]
            return format(metrics, row, col)
        grid = np.array([
            [
                ['train_loss', 'valid_loss'],
                ['val_acc', 'val_recall', 'val_prec'],
            ],
            [
                ['pred_train_loss', 'pred_val_loss'],
                ['cls_train_loss', 'cls_val_loss'],
            ],
        ])
        util.show_metrics(data, grid, title_format=title_format)


if __name__ == "__main__":
    LogAnalyser.main(None)
