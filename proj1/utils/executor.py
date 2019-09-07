# %%
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# %%


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

    def epoch_reset(self):
        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.valid_acc = []

    def epoch_step(self, epoch_idx):
        self.curr_epoch_id += 1
        self.curr_epoch_idx = epoch_idx
        self.curr_epoch_title = '{}/{}'.format(epoch_idx, self.args.epochs)

    def epoch_loss_report(self, train_loss, valid_loss, train_acc, valid_acc):
        epoch_id = self.curr_epoch_id
        executor = self.executor

        self.log('Train: loss: {:.6f}'.format(train_loss))
        self.log('Train: acc: {:.6f}'.format(train_acc))
        self.log('Valid: loss: {:.6f}'.format(valid_loss))
        self.log('Valid: acc: {:.6f}'.format(valid_acc))
        self.log('====================================================')
        self.execlog(
            train_loss=train_loss,
            valid_loss=valid_loss,
            train_acc=train_acc,
            valid_acc=valid_acc,
            tag='loss',
            epoch=epoch_id,
        )

        if np.isnan(train_loss) or np.isnan(valid_loss):
            return

        if self.args.save_model:
            model_filepath = os.path.join(
                executor.exec_save_dir, 'epoch_{}.pt'.format(epoch_id))

            torch.save(executor.model.state_dict(), model_filepath)

        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.train_acc.append(train_acc)
        self.valid_acc.append(valid_acc)
        self.draw_epoch_loss()

    def draw_epoch_loss(self):
        train_losses = self.train_losses
        valid_losses = self.valid_losses
        train_acc = self.train_acc
        valid_acc = self.valid_acc

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

        plt.show(block=False)
        plt.pause(0.000001)

    def report_batch_loss(self, num_batch_samples, num_samples, total_samples, num_batch, total_batch, loss, corrected):
        """
        num_samples: 已经训练的样本数  
        total_samples: 总样本数  
        num_batch: 已经训练的`batch`数  
        total_batch: 总`batch`数  
        """
        if num_batch % self.args.log_interval == 0:
            self.log(
                'Train Epoch: {} [{}/{} ({}/{})]\t loss: {:.6f}\t corrected: {}/{}'.format(
                    self.curr_epoch_title,
                    num_samples,
                    total_samples,
                    num_batch,
                    total_batch,
                    loss,
                    corrected,
                    num_batch_samples,
                )
            )


class Executor():
    def __init__(self, args):
        from utils import shortuuid
        self.execid = shortuuid.uuid()

        self._init_args(args)
        self._make_dirs_if_need()

        self._init_log(args)
        self._init_tracer(args)
        self._init_device(args)

        self.log('====> Loading Datasets')
        self._init_data_loader(args)

        self.log("====> Building Model")
        self._init_net_model(args)

        self._init_criterion(args)
        self._init_optimizer(args)
        self._init_lr_scheduler(args)

    def _init_args(self, args):
        self.args = args
        args.phase = args.phase.lower()
        self.exec_save_dir = os.path.join(args.save_directory, self.execid)

    def _make_dirs_if_need(self):
        # 模型、日志目录
        save_directory = self.args.save_directory
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 本次运行的模型、日志子目录
        if not os.path.exists(self.exec_save_dir):
            os.makedirs(self.exec_save_dir)

    def _init_log(self, args):
        """
        日志初始化:
        log: 控制台输出
        rootlog: 文件输出、根目录下的日志文件、记录本次执行的参数
        execlog: 文件输出、`execid`子目录日志文件、记录本次执行的详细日志
        """
        from utils import logutil
        std_logger = logutil.std_logger(
            'std', '[%(asctime)s] %(message)s')
        self.log = std_logger.info

        filepath = os.path.join(args.save_directory, 'log.log')
        root_logger = logutil.file_logger('file.root', filepath)
        self.rootlog = logutil.get_struct_log(
            root_logger.info, execid=self.execid, phase=args.phase)

        exec_log_file = os.path.join(self.exec_save_dir, 'log.log')
        exec_logger = logutil.file_logger('file.exec', exec_log_file)
        self.execlog = logutil.get_struct_log(exec_logger.info)

        self.rootlog(args=args)

    def _init_tracer(self, args):
        self.tracer = LossTracer(args, self)

    def _init_device(self, args):
        """
        设备初始化: GPU or CPU
        `args.no_cuda`
        """
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def _init_data_loader(self, args):
        """
        加载数据集: 测试集、验证集
        """
        raise NotImplementedError

    def _init_net_model(self, args):
        """
        网络模型
        """
        raise NotImplementedError

    def _init_criterion(self, args):
        """
        损失函数
        """
        raise NotImplementedError

    def _init_optimizer(self, args):
        """
        优化器
        """
        raise NotImplementedError

    def _init_lr_scheduler(self, args):
        """
        学习策略
        """
        raise NotImplementedError

    def _train_phase_(self):
        raise NotImplementedError

    def run(self):
        phase = self.args.phase

        fn_name = '_{}_phase_'.format(phase)
        fn = self.__getattribute__(fn_name)
        if fn is None:
            raise Exception('unknown phase: {}'.format(phase))
        fn()

    @classmethod
    def get_phases(self):
        import re

        regexp = re.compile(r'_([\w]+)_phase_')

        def get_phase(name):
            result = regexp.findall(name)
            return result[0] if result else None

        return filter(lambda x: x is not None, map(get_phase, dir(self)))

    @classmethod
    def get_args_parser(self, description, args=None):
        from utils.util import get_args_parser

        parser, arg = get_args_parser(description)

        arg('--phase', type=str, default='train',
            metavar='|'.join(self.get_phases()),
            help='phase to run')

        arg('--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')

        arg('--test-batch-size', type=int, default=64, metavar='N',
            help='input batch size for testing (default: 64)')

        arg('--epochs', type=int, default=100, metavar='N',
            help='number of epochs to train (default: 100)')

        arg('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')

        arg('--momentum', type=float, default=0.5, metavar='M',
            help='SGD momentum (default: 0.5)')

        arg('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')

        arg('--seed', type=int, default=1,
            metavar='Seed',
            help='random seed (default: 1)')

        arg('--log-interval', type=int, default=5, metavar='N',
            help='how many batches to wait before logging training status')

        arg('--save-model', action='store_true', default=True,
            help='save the current Model')

        arg('--save-directory', type=str, default='out',
            help='learnt models and logs are saving here')

        arg('--data-directory', type=str,
            help='data are loading from here')

        arg('--model-file', type=str,
            help='model are loading from here')

        arg('--predict-indices', type=str, default='all',
            help='sample indices to predict')

        arg('--no-cache-image', action='store_true', default=False,
            help='should cache image in memory')

        arg('--retry', action='store_true', default=True,
            help='loss为nan时是否自动需要调整参数重试')

        return parser, arg
