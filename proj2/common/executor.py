# %%
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from . import util
# %%


def retry(func):
    def wrapper(self):
        args = self.args
        if not args.retry:
            return func(self)

        if not hasattr(self, 'retry_id'):
            self.retry_id = 0
        else:
            self.retry_id = self.retry_id + 1
        if func(self):
            args.model_file = self.random_previous_model()
            args.lr = args.lr / math.sqrt(3)
            self.reset_model()

            self.log(
                'retry: lr: {}, model: {}'.format(args.lr, args.model_file))
            self.execlog(
                lr=args.lr, model_file=args.model_file,
                tag='retry', epoch=self.tracer.curr_epoch_id,
            )
            return wrapper(self)

    return wrapper


class Executor():
    def __init__(self, args):
        from . import shortuuid
        self.execid = shortuuid.uuid()
        self.retry_id = 0

        self._init_args(args)
        self._make_dirs_if_need()

        self._init_log()
        self._init_tracer()
        self._init_device()

        self.log('====> Loading Datasets')
        self._init_data_loader()

        self.log("====> Building Model")
        self._init_net_model()

        self._init_criterion()
        self._init_optimizer()
        self._init_lr_scheduler()

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

    def _init_log(self):
        """
        日志初始化:
        log: 控制台输出
        rootlog: 文件输出、根目录下的日志文件、记录本次执行的参数
        execlog: 文件输出、`execid`子目录日志文件、记录本次执行的详细日志
        """
        args = self.args
        from . import logutil
        std_logger = logutil.std_logger(
            'std', '[%(asctime)s] %(message)s')
        self.log = std_logger.info

        if self.should_make_exec_save_dir():

            filepath = os.path.join(args.save_directory, 'log.log')
            root_logger = logutil.file_logger('file.root', filepath)
            self.rootlog = logutil.get_struct_log(
                root_logger.info, execid=self.execid, phase=args.phase)

            exec_log_file = os.path.join(self.exec_save_dir, 'log.log')
            exec_logger = logutil.file_logger('file.exec', exec_log_file)
            self.execlog = logutil.get_struct_log(exec_logger.info)
        else:
            def nop(*args, **kwargs):
                pass
            self.rootlog = nop
            self.execlog = nop
        self.rootlog(args=args)

    def _init_tracer(self):
        args = self.args
        from .tracer import Tracer
        self.tracer = Tracer(args, self)

    def _init_device(self):
        """
        设备初始化: GPU or CPU
        `args.no_cuda`
        """
        args = self.args
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def get_dataset_class(self):
        from . import dataset
        return dataset.FaceLandmarksDataset

    def _init_data_loader(self):
        """
        加载数据集: 测试集、验证集
        """
        args = self.args
        Dataset = self.get_dataset_class()

        train_set, test_set = Dataset.get_train_test_set(
            args)

        self.train_data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
        self.valid_data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.test_batch_size)

    def _import_network(self):
        raise NotImplementedError

    def _init_net_model(self):
        """
        网络模型
        """
        args = self.args
        network = self._import_network()
        name = args.model
        if name != 'Net':
            name = 'Net_{}'.format(name)

        Net = getattr(network, name) if hasattr(network, name) else network.Net

        self.model = Net().to(self.device)

        if self.args.model_file is None:
            self.init_parameters()
        else:
            self.load_state_dict()

    def init_parameters(self):
        """
        初始化参数，默认使用`kaiming`，子类可以重写为其他方式
        """
        self.init_parameters_kaiming()

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

    def reset_model(self):
        self._init_net_model()
        self._init_optimizer()

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

    def _init_criterion(self):
        """
        损失函数
        """
        raise NotImplementedError

    def _init_optimizer(self):
        """
        优化器
        """
        raise NotImplementedError

    def _init_lr_scheduler(self):
        """
        学习策略
        """
        raise NotImplementedError

    def _train_phase_(self):
        raise NotImplementedError

    def should_make_exec_save_dir(self):
        return self.args.phase == 'train'

    def random_previous_model(self):
        import random
        models = filter(lambda x: x.startswith('epoch_'),
                        os.listdir(self.exec_save_dir))
        sorted_models = sorted(models, key=lambda x: int(x[6:-3]))

        offset = min(random.randrange(1, 6), len(sorted_models))
        if offset > 0:
            model_name = sorted_models[-offset]
            return os.path.join(self.exec_save_dir, model_name)
        return None

    def show_trainset(self):
        Dataset = self.get_dataset_class()

        train_set, _ = Dataset.get_train_test_set(
            self.args)
        Dataset.show_dataset(train_set)

    def show_testset(self):
        Dataset = self.get_dataset_class()

        _, test_set = Dataset.get_train_test_set(
            self.args)
        Dataset.show_dataset(test_set)

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
        util.get_action_fn(self, self.args.phase, 'phase')()
        plt.show()

    @classmethod
    def get_phases(self):
        return util.get_actions(self, 'phase')

    @classmethod
    def get_args_parser(self, description, args=None):
        from .util import get_args_parser

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

        arg('--lr', type=float, default=0.001, metavar='LR',
            help='learning rate (default: 0.001)')

        arg('--momentum', type=float, default=0.5, metavar='M',
            help='SGD momentum (default: 0.5)')

        arg('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')

        arg('--seed', type=int, default=1,
            metavar='Seed',
            help='random seed (default: 1)')

        arg('--log-interval', type=int, default=20, metavar='N',
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
