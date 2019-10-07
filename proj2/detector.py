#!/usr/bin/env python
# %%
from data import get_train_test_set
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import math
from loss_tracer import LossTracer

# %%

class Detector():

    def __init__(self, args):
        import shortuuid
        self.execid = shortuuid.uuid()

        self.next_retry_id = 0

        self._init_args(args)
        self._make_dirs_if_need()

        self._init_log(args)
        self._init_tracer(args)

        self._init_device(args)
        self.log("====> Loading Datasets")
        self._init_data_loader(args)
        self.log("====> Building Model")
        self._init_net_model(self.device)
        self._init_criterion(args)
        self._init_optimizer(args, self.model)
        self._init_lr_scheduler(args)

    def _init_args(self, args):
        """
        参数预处理
        """
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
        import logutil
        std_logger = logutil.std_logger(
            'detector.std', '[%(asctime)s] %(message)s')
        self.log = std_logger.info

        filepath = os.path.join(args.save_directory, 'log.log')
        root_logger = logutil.file_logger('detector.root', filepath)
        self.rootlog = logutil.get_struct_log(
            root_logger.info, execid=self.execid, phase=args.phase)

        if self.should_make_exec_save_dir():
            exec_log_file = os.path.join(self.exec_save_dir, 'log.log')
            exec_logger = logutil.file_logger('detector.exec', exec_log_file)
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

    def _init_data_loader(self, args):
        """
        加载数据集: 测试集、验证集
        """
        train_set, test_set = get_train_test_set(
            args.data_directory, not args.no_cache_image)
        self.train_data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
        self.valid_data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.test_batch_size)

    def _init_net_model(self, device):
        """
        网络模型
        """
        import network
        name = args.model
        if name != 'Net':
            name = 'Net_{}'.format(name)

        Net = getattr(network, name) if hasattr(network, name) else network.Net

        self.model = Net().to(device)

        if self.args.model_file is None:
            self.init_parameters_kaiming()
        else:
            self.load_state_dict()

    def init_parameters_kaiming(self):
        """
        使用`kaiming_normal_`初始化参数
        """
        for p in self.model.parameters():
            if len(p.shape) >= 2:
                nn.init.kaiming_normal_(p)

    def _init_criterion(self, args):
        """
        损失函数
        """
        self.criterion = nn.MSELoss()

    def _init_optimizer(self, args, model):
        """
        优化器
        """
        self.optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)

    def _init_lr_scheduler(self, args):
        """
        学习策略
        """
        self.lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.1)

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

    def reset_model(self):
        self._init_net_model(self.device)
        self._init_optimizer(self.args, self.model)

    def draw_losses(self, train_losses, valid_losses):
        plt.clf()
        plt.plot(range(len(train_losses)), train_losses)
        plt.plot(range(len(valid_losses)), valid_losses)
        plt.legend(['train_losses', 'valid_losses'])
        plt.show(block=False)
        plt.pause(0.000001)

    def run(self):
        """
        执行阶段
        """
        fn_name = '{}_phase'.format(self.args.phase)
        fn = self.__getattribute__(fn_name)
        if fn is not None:
            fn()

    def run_an_epoch(self, is_training=True, data_loader = None):
        tracer = self.tracer

        if data_loader is None:
            data_loader = self.train_data_loader if is_training else self.valid_data_loader
        total_samples = len(data_loader.dataset)
        total_batch = len(data_loader)

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        num_samples = 0

        for batch_idx, batch in enumerate(data_loader):
            inputs = batch['image'].to(self.device)
            ground_truth = batch['landmarks'].to(self.device)

            if is_training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):
                outputs = self.model(inputs)

                loss = self.criterion(outputs, ground_truth)

                if is_training:
                    loss.backward()
                    self.optimizer.step()

            num_batch_samples = len(inputs)
            num_samples += num_batch_samples
            loss_value = loss.item()
            running_loss += loss_value

            tracer.report_batch_loss(
                num_batch_samples, num_samples, total_samples, batch_idx + 1, total_batch, loss_value)

        running_loss /= total_batch * 1.0
        return running_loss

    def should_make_exec_save_dir(self):
        return self.args.phase == 'train'

    def _train_phase(self):
        self.log('====> Start Training')
        args = self.args
        tracer = self.tracer

        retry_id = self.next_retry_id
        self.next_retry_id = self.next_retry_id + 1

        tracer.epoch_reset()

        for idx in range(args.epochs):
            tracer.epoch_step(idx)

            train_loss = self.run_an_epoch(True)
            valid_loss = self.run_an_epoch(False)
    
            if np.isnan(train_loss) or np.isnan(valid_loss):
                if args.retry:
                    args.model_file = self.random_previous_model()
                    args.lr = args.lr / math.sqrt(3)
                    self.reset_model()

                    self.log(
                        'retry: lr: {}, model: {}'.format(args.lr, args.model_file))
                    self.execlog(
                        lr=args.lr, model_file=args.model_file,
                        tag='retry', epoch=tracer.curr_epoch_id,
                    )
                return args.retry
            else:
                tracer.epoch_loss_report({
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                }, retry_id)

        return False

    def train_phase(self):
        while self._train_phase():
            pass
        # plt.show()

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

    def test_phase(self):
        self.log('====> Testing Model on the train set')

        train_loss = self.run_an_epoch(False, self.train_data_loader)
        self.log('loss for the train set: {}'.format(train_loss))
        self.execlog(train_loss=train_loss)

        self.log('====> Testing Model on the test set')
        test_loss = self.run_an_epoch(False)
        self.log('loss for the test set: {}'.format(test_loss))
        self.execlog(test_loss=test_loss)

    def finetune_phase(self):
        # self.train_phase()
        pass

    def get_predict_dataset(self):
        tokens = self.args.predict_indices.split('@')

        predict_indices = tokens[0]
        dataset_name = tokens[1] if len(tokens) > 1 else 'test'

        dataset = self.valid_data_loader.dataset if dataset_name.lower(
        ) == 'test' else self.train_data_loader.dataset

        if predict_indices == 'all':
            indices = range(len(dataset))
        else:
            indices = map(int, filter(lambda x: len(
                x) > 0, predict_indices.split(',')))
        return dataset, indices

    def predict_phase(self):
        dataset, indices = self.get_predict_dataset()

        self.model.eval()
        with torch.no_grad():
            for idx in indices:
                if idx < 0 or idx >= len(dataset):
                    continue
                img = dataset[idx]['image']
                landmarks = dataset[idx]['landmarks']
                image_name = dataset[idx]['image_name']

                input_img = img.expand(1, *img.size())

                output = self.model(input_img)[0]

                self.log('current image: {}'.format(image_name))
                plt.title(image_name)
                plt.imshow(img[0])
                plt.scatter(landmarks[0::2], landmarks[1::2],
                            alpha=0.5, color='yellow')

                plt.scatter(output[0::2], output[1::2],
                            alpha=0.5, color='r')

                plt.show()


def _parse_args():
    from util import get_args_parser

    parser, p = get_args_parser('Detector')

    p('--batch-size', type=int, default=64, metavar='N',
        help='input batch size for training (default: 64)')
    p('--test-batch-size', type=int, default=64, metavar='N',
        help='input batch size for testing (default: 64)')
    p('--epochs', type=int, default=100, metavar='N',
        help='number of epochs to train (default: 100)')
    p('--lr', type=float, default=0.001, metavar='LR',
        help='learning rate (default: 0.00003)')
    p('--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)')
    p('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
    p('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    p('--log-interval', type=int, default=20, metavar='N',
        help='how many batches to wait before logging training status')
    p('--save-model', action='store_true', default=True,
        help='save the current Model')
    p('--model', type=str, default='Net',
        help='model name')

    p('--save-directory', type=str, default='out',
        help='learnt models are saving here')
    p('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
        help='training, predicting or finetuning')
    p('--data-directory', type=str,
        help='data are loading from here')

    p('--model-file', type=str,
        help='model are loading from here')

    p('--predict-indices', type=str, default='all',
        help='sample indices to predict')

    p('--no-cache-image', action='store_true', default=False,
        help='should cache image in memory')

    p('--retry', action='store_true', default=True,
        help='loss为nan时是否自动需要调整参数重试')

    args = parser.parse_args()

    return args


# %%
if __name__ == "__main__":
    args = _parse_args()

    # 随机种子
    torch.manual_seed(args.seed)

    detector = Detector(args)
    detector.run()
