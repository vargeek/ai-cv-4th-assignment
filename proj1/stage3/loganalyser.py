#!/usr/bin/env python
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

IPYTHON_MODE = 'get_ipython' in dir()
CUR_DIR = os.path.curdir if IPYTHON_MODE else os.path.dirname(
    __file__)
sys.path.append(os.path.join(CUR_DIR, '..'))

# from utils import LogAnalyser

def _importLogAnalyser():
    from utils import LogAnalyser
    return LogAnalyser

class LogAnalyser2(_importLogAnalyser()):
    @staticmethod
    def main(args):
        parser, _ = LogAnalyser2.get_args_parser()
        LogAnalyser2(parser.parse_args(args)).run()

    def show_train_loss(self):
        data = self.load_execlog_df('train', 'loss')

        losses_keys = [
            'train_loss',
            'valid_loss',
        ]

        accs_keys = [
            'train_acc_c',
            'val_acc_c',
            'train_acc_s',
            'val_acc_s',
            'val_acc',
        ]

        from utils.util import show_losses_accs
        losses = {
            k: data[k] for k in losses_keys
        }
        accs = {
            k: data[k] for k in accs_keys
        }

        show_losses_accs(losses, accs)

    def get_best_val(self):
        data = self.load_execlog_df('train', 'loss')

        best_loss_epoch = None
        best_acc_epoch = None
        best_loss = None
        best_acc = None
        best_acc_s = None
        best_acc_c = None
        best_acc_s_epoch = None
        best_acc_c_epoch = None
        for i in range(len(data)):
            item = data.iloc[i]
            epoch = item['epoch']
            valid_loss = item['valid_loss']
            valid_acc = item['val_acc']
            valid_acc_s = item['val_acc_s']
            valid_acc_c = item['val_acc_c']

            if best_loss is None or valid_loss < best_loss:
                best_loss = valid_loss
                best_loss_epoch = epoch

            if best_acc is None or valid_acc > best_acc:
                best_acc = valid_acc
                best_acc_epoch = epoch

            if best_acc_s is None or valid_acc_s > best_acc_s:
                best_acc_s = valid_acc_s
                best_acc_s_epoch = epoch

            if best_acc_c is None or valid_acc_c > best_acc_c:
                best_acc_c = valid_acc_c
                best_acc_c_epoch = epoch

        print('best valid loss: epoch: {}, loss: {}'.format(
            best_loss_epoch, best_loss))
        print('best valid acc: epoch: {}, acc: {}'.format(
            best_acc_epoch, best_acc))
        print('best valid acc_s: epoch: {}, acc: {}'.format(
            best_acc_s_epoch, best_acc_s))
        print('best valid acc_c: epoch: {}, acc: {}'.format(
            best_acc_c_epoch, best_acc_c))

if __name__ == "__main__":

    LogAnalyser2.main([] if IPYTHON_MODE else None)
