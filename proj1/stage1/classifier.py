#!/usr/bin/env python
# %%
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

IPYTHON_MODE = 'get_ipython' in dir()
CUR_DIR = os.path.curdir if IPYTHON_MODE else os.path.dirname(
    __file__)
sys.path.append(os.path.join(CUR_DIR, '..'))


def importExecutor():
    from utils import Executor
    return Executor


class Classifier(importExecutor()):
    def __init__(self, args):
        super(Classifier, self).__init__(args)
        self.next_epoch_id = 0

    def _init_data_loader(self, args):
        """
        加载数据集: 测试集、验证集
        """
        from dataset import get_train_test_set
        train_set, test_set = get_train_test_set(
            args)

        self.train_data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
        self.valid_data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.test_batch_size)

    def _init_net_model(self, args):
        """
        网络模型
        """
        from Classes_Network import Net
        self.model = Net().to(self.device)
        self.load_state_dict()

    def _init_criterion(self, args):
        """
        损失函数
        """
        # self.criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()

    def _init_optimizer(self, args):
        """
        优化器
        """
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=args.lr, momentum=args.momentum)

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

    def train_or_validate_an_epoch(self, is_training=True):
        tracer = self.tracer

        data_loader = self.train_data_loader if is_training else self.valid_data_loader
        total_samples = len(data_loader.dataset)
        total_batch = len(data_loader)

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        num_corrected = 0
        num_samples = 0

        for batch_idx, batch in enumerate(data_loader):
            inputs = batch['image'].to(self.device)
            ground_truth = batch['classes'].to(self.device)

            if is_training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):
                outputs = self.model(inputs)

                _, predicted = torch.max(outputs, 1)

                loss = self.criterion(outputs, ground_truth)

                if is_training:
                    loss.backward()
                    self.optimizer.step()

            num_batch_samples = len(inputs)
            num_samples += num_batch_samples
            loss_value = loss.item()
            running_loss += loss_value

            corrected = torch.sum(predicted == ground_truth).item()
            num_corrected += corrected

            tracer.report_batch_loss(
                num_batch_samples, num_samples, total_samples, batch_idx + 1, total_batch, loss_value, corrected)

        running_loss /= total_batch * 1.0
        running_acc = num_corrected * 1.0 / total_samples
        return running_loss, running_acc

    # phases
    def _train_phase_(self):
        self.log('====> Start Training')
        tracer = self.tracer

        tracer.epoch_reset()
        for idx in range(args.epochs):
            tracer.epoch_step(idx)

            train_loss, train_acc = self.train_or_validate_an_epoch(True)
            valid_loss, valid_acc = self.train_or_validate_an_epoch(False)

            tracer.epoch_loss_report(
                train_loss, valid_loss, train_acc, valid_acc)

    def _test_phase_(self):
        print('xxx')

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


def _get_args(args=None):
    parser, _ = Classifier.get_args_parser('Classifier')
    return parser.parse_args(args)


if __name__ == "__main__":
    args = _get_args([] if IPYTHON_MODE else None)

    # 随机种子
    torch.manual_seed(args.seed)

    classifier = Classifier(args)
    classifier.run()


# %%
