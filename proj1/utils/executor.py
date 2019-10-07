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
        self.min_valid_loss = None
        self.max_valid_acc = None

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
        self.max_valid_acc = valid_acc if self.max_valid_acc is None else max(self.max_valid_acc, valid_acc)
        self.min_valid_loss = valid_loss if self.min_valid_loss is None else min(self.min_valid_loss, valid_loss)
        
        self.draw_epoch_loss()

    def draw_epoch_loss(self):

        from utils.util import show_train_loss

        show_train_loss(self.train_losses, self.valid_losses, self.train_acc, self.valid_acc, self.max_valid_acc, self.min_valid_loss, block=False)

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

            # self.log(self.executor.optimizer.param_groups)


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
        if self.should_make_exec_save_dir() and not os.path.exists(self.exec_save_dir):
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

        if self.should_make_exec_save_dir():
            exec_log_file = os.path.join(self.exec_save_dir, 'log.log')
            exec_logger = logutil.file_logger('file.exec', exec_log_file)
            self.execlog = logutil.get_struct_log(exec_logger.info)
        else:
            def nop(*args, **kwargs):
                pass
            self.execlog = nop
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

    def _import_dataset(self):
        raise NotImplementedError

    def _init_data_loader(self, args):
        """
        加载数据集: 测试集、验证集
        """
        dataset = self._import_dataset()

        train_set, test_set = dataset.get_train_test_set(
            args)

        self.train_data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
        self.valid_data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.test_batch_size)

    def _init_net_model(self, args):
        """
        网络模型
        """
        self.model = None
        raise NotImplementedError

    def init_parameters_kaiming(self):
        """
        使用`kaiming_normal_`初始化参数
        """
        for p in self.model.parameters():
            if len(p.shape) >= 2:
                nn.init.kaiming_normal_(p)

    def init_parameters_xavier(self):
        """
        使用`xavier_normal_`初始化参数
        """
        for p in self.model.parameters():
            nn.init.xavier_normal_(p)

    def load_state_dict(self):
        """
        加载模型
        """
        filepath = self.args.model_file
        if filepath is None:
            return

        self.log("====> Loading Model: {}".format(filepath))
        self.execlog(model_file=filepath, tag='load')
        state_dict = torch.load(filepath)
        self.model.load_state_dict(state_dict)


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

    def should_make_exec_save_dir(self):
        return self.args.phase == 'train'

    def get_predict_dataset(self):
        tokens = self.args.predict_indices.split('@')

        predict_indices = tokens[0]
        dataset_name = tokens[1] if len(tokens) > 1 else 'val'

        dataset = self.valid_data_loader.dataset if dataset_name.lower(
        ) == 'val' else self.train_data_loader.dataset

        if predict_indices == 'all':
            indices = range(len(dataset))
        else:
            indices = map(int, filter(lambda x: len(
                x) > 0, predict_indices.split(',')))
        return dataset, indices

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

        arg('--model', type=str, default='Net',
            help='model name')

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
