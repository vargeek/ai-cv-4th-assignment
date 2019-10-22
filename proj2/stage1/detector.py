#!/usr/bin/env python
import init
from common import executor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt


class Detector(executor.Executor):
    def __init__(self, args):
        super(Detector, self).__init__(args)
        self.retry_id = 0

    def _import_network(self):
        import network
        return network

    def _init_tracer(self):
        super(Detector, self)._init_tracer()

        from common.util import title_format_for_fields
        self.tracer.title_format = title_format_for_fields(['valid_loss'])

    def _init_criterion(self):
        """
        损失函数
        """
        self.criterion = nn.MSELoss()

    def _init_optimizer(self):
        """
        优化器
        """
        args = self.args
        model = self.model
        self.optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)

    def _init_lr_scheduler(self):
        """
        学习策略
        """
        self.lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.1)

    def run_an_epoch(self, is_training=True, data_loader=None):
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

            tracer.batch_report(num_samples, total_samples,
                                batch_idx+1, total_batch, loss=loss_value)

        running_loss /= total_batch * 1.0
        return running_loss

    @executor.retry
    def _train_phase_(self):
        self.log('====> Start Training')
        args = self.args
        tracer = self.tracer

        tracer.epoch_reset()

        for idx in range(args.epochs):
            tracer.epoch_step(idx)

            train_loss = self.run_an_epoch(True)
            valid_loss = self.run_an_epoch(False)

            if np.isnan(train_loss) or np.isnan(valid_loss):
                return True

            tracer.epoch_report([
                ('train_loss', train_loss), ('valid_loss', valid_loss),
            ], retry_id=self.retry_id)

        return False

    def _test_phase_(self):
        self.log('====> Testing Model on the train set')

        train_loss = self.run_an_epoch(False, self.train_data_loader)
        self.log('loss for the train set: {}'.format(train_loss))
        self.execlog(train_loss=train_loss)

        self.log('====> Testing Model on the test set')
        test_loss = self.run_an_epoch(False)
        self.log('loss for the test set: {}'.format(test_loss))
        self.execlog(test_loss=test_loss)

    def _predict_phase_(self):
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


if __name__ == "__main__":
    # 随机种子

    parser, arg = Detector.get_args_parser('detector')
    args = parser.parse_args([] if
                             init.is_ipython_mode() else None)
    args.stage = init.STAGE

    torch.manual_seed(args.seed)

    exec = Detector(args)
    exec.run()
