
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class LossTracer():
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
        self.losses = {}

    def epoch_reset(self):
        self.losses = {}

    def _get_loss_list(self, key):
        loss = self.losses.get(key)
        if loss is None:
            loss = []
            self.losses[key] = loss
        return loss

    def epoch_step(self, epoch_idx):
        self.curr_epoch_id += 1
        self.curr_epoch_idx = epoch_idx
        self.curr_epoch_title = '{}/{}'.format(epoch_idx, self.args.epochs)

    def epoch_loss_report(self, losses, retry_id):
        epoch_id = self.curr_epoch_id
        executor = self.executor

        for k,v in losses.items():
            self.log('{}: {:.6f}'.format(k, v))
        
        self.log('====================================================')
        self.execlog(
            **losses,
            tag='loss',
            epoch=epoch_id,
            retry_id = retry_id,
        )

        for k,v in losses.items():
            if np.isnan(v):
                return

        if self.args.save_model:
            model_filepath = os.path.join(
                executor.exec_save_dir, 'epoch_{}.pt'.format(epoch_id))

            torch.save(executor.model.state_dict(), model_filepath)


        for k,v in losses.items():
            self._get_loss_list(k).append(v)
        
        self.draw_epoch_loss()

    def draw_epoch_loss(self):
        from util import draw_losses
        draw_losses(self.losses, block=False)

    def report_batch_loss(self, num_batch_samples, num_samples, total_samples, num_batch, total_batch, loss):
        """
        num_samples: 已经训练的样本数  
        total_samples: 总样本数  
        num_batch: 已经训练的`batch`数  
        total_batch: 总`batch`数  
        """
        if num_batch % self.args.log_interval == 0:
            self.log(
                'Train Epoch: {} [{}/{} ({}/{})]\t loss: {:.6f}'.format(
                    self.curr_epoch_title,
                    num_samples,
                    total_samples,
                    num_batch,
                    total_batch,
                    loss,
                )
            )

            # self.log(self.executor.optimizer.param_groups)

