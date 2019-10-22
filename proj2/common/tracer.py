import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from common import util


class Tracer():
    def __init__(self, args, executor):
        """
        args: {log_interval, epochs}
        """
        self.args = args
        self.executor = executor
        self.curr_epoch_id = 0
        self.curr_epoch_idx = 0
        self.curr_epoch_title = ''
        self.log = executor.log
        self.execlog = executor.execlog
        self.metrics = {}
        # title_format(metrics, row, col) -> str
        self.title_format = None

    def epoch_reset(self):
        self.metrics = {}

    def epoch_step(self, epoch_idx):
        self.curr_epoch_id += 1
        self.curr_epoch_idx = epoch_idx
        self.curr_epoch_title = '{}/{}'.format(epoch_idx, self.args.epochs)

    def epoch_report(self, metrics, grid=None, **info):
        epoch_id = self.curr_epoch_id
        executor = self.executor
        for (key, value) in metrics:
            self.log('{}: {:.6f}'.format(key, value))
            metric = self.metrics.get(key)
            if metric is None:
                metric = []
                self.metrics[key] = metric
            metric.append(value)

        self.log('====================================================')
        self.execlog(
            **dict(metrics),
            tag='loss',
            epoch=epoch_id,
            **info,
        )

        if self.args.save_model:
            model_filepath = os.path.join(
                executor.exec_save_dir, 'epoch_{}.pt'.format(epoch_id))

            torch.save(executor.model.state_dict(), model_filepath)

        self.draw_epoch_metrics(grid)

    def draw_epoch_metrics(self, grid=None):
        from .util import show_metrics
        if grid is None:
            grid = list(self.metrics.keys())
        show_metrics(self.metrics, grid,
                     block=False, title_format=self.title_format)

    def batch_report(self, num_samples, total_samples, num_batch, total_batch, **metrics):
        """
        num_samples: 已经训练的样本数  
        total_samples: 总样本数  
        num_batch: 已经训练的`batch`数  
        total_batch: 总`batch`数  
        """
        if num_batch % self.args.log_interval == 0:
            msg = 'Train Epoch: {} [{}/{} ({}/{})]'.format(
                self.curr_epoch_title,
                num_samples,
                total_samples,
                num_batch,
                total_batch,
            )
            for k, v in metrics.items():
                msg = msg + '\t {}: {}'.format(k, v)

            self.log(msg)
