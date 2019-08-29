#!/usr/bin/env python
# %%
from data import get_train_test_set
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import shortuuid
import math

# %%


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        avgPool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 1*112*112 -> 8*54*54
        self.conv1_1 = nn.Conv2d(1, 8, kernel_size=5, stride=2)
        self.prelu1_1 = nn.PReLU()
        # 8*54*54 -> 8*27*27
        self.pool1 = avgPool

        # 8*54*54 -> 16*25*25
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3)
        self.prelu2_1 = nn.PReLU()
        # 16*25*25 -> 16*23*23
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3)
        self.prelu2_2 = nn.PReLU()
        # 16*23*23 -> 16*12*12
        self.pool2 = avgPool

        # 16*12*12 -> 24*10*10
        self.conv3_1 = nn.Conv2d(16, 24, kernel_size=3)
        self.prelu3_1 = nn.PReLU()
        # 24*10*10 -> 24*8*8
        self.conv3_2 = nn.Conv2d(24, 24, kernel_size=3)
        self.prelu3_2 = nn.PReLU()
        # 24*8*8 -> 24*4*4
        self.pool3 = avgPool

        # 24*4*4 -> 40*4*4
        self.conv4_1 = nn.Conv2d(24, 40, kernel_size=3, padding=1)
        self.prelu4_1 = nn.PReLU()
        # 40*4*4 -> 80*4*4
        self.conv4_2 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.prelu4_2 = nn.PReLU()

        self.ip1 = nn.Linear(80 * 4 * 4, 128)
        self.preluip1 = nn.PReLU()
        self.ip2 = nn.Linear(128, 128)
        self.preluip2 = nn.PReLU()
        self.ip3 = nn.Linear(128, 42)

    def forward(self, x):
        """
        x: (1,1,112,112)
        retVal: (1, 42)
        """
        x = self.prelu1_1(self.conv1_1(x))
        x = self.pool1(x)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = self.pool2(x)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = self.pool3(x)

        x = self.prelu4_1(self.conv4_1(x))
        x = self.prelu4_2(self.conv4_2(x))

        x = x.view(-1, 80 * 4 * 4)
        x = self.preluip1(self.ip1(x))

        x = self.preluip2(self.ip2(x))
        x = self.ip3(x)

        return x


class Detector():

    def __init__(self, args):
        self.uuid = shortuuid.uuid()
        self.next_epoch_id = 0
        self.next_retry_id = 0

        self._init_args(args)
        self._make_dirs_if_need()

        self._init_log(args)

        self._init_device(args)
        self.log("====> Loading Datasets")
        self._init_data_loader(args)
        self.log("====> Building Model")
        self._init_net_model(self.device)
        self._init_criterion(args)
        self._init_optimizer(args, self.model)

    def _init_args(self, args):
        """
        参数预处理
        """
        self.args = args

        args.phase = args.phase.lower()

        self.save_subdir = os.path.join(args.save_directory, self.uuid)

    def _make_dirs_if_need(self):
        # 模型、日志目录
        save_directory = args.save_directory
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 本次运行的模型、日志子目录
        if not os.path.exists(self.save_subdir):
            os.makedirs(self.save_subdir)

    def _init_log(self, args):
        import logutil
        std_logger = logutil.std_logger(
            'detector.std', '[%(asctime)s] %(message)s')
        self.log = std_logger.info

        filepath = os.path.join(args.save_directory, 'log.log')
        root_logger = logutil.file_logger('detector.root', filepath)
        self.rootlog = logutil.get_struct_log(
            root_logger.info, uuid=self.uuid, phase=args.phase)

        subfilepath = os.path.join(self.save_subdir, 'log.log')
        sub_logger = logutil.file_logger('detector.sub', subfilepath)
        # self.sublog = logutil.get_struct_log(sub_logger.info, phase=args.phase)
        self.sublog = logutil.get_struct_log(sub_logger.info)

        self.rootlog(args=args)

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
        self.model = Net().to(device)
        self.load_state_dict()

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

    def load_state_dict(self):
        """
        加载模型
        """
        filepath = self.args.model_file
        if filepath is None:
            return

        self.log("====> Loading Model: {}".format(filepath))
        self.sublog(model_file=filepath, tag='load')
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

    def do_train(self, epoch_title=''):
        args = self.args
        data_loader = self.train_data_loader
        num_dataset = len(data_loader.dataset)
        num_batch = len(data_loader)

        # Sets the module in training mode
        self.model.train()

        running_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            imgs = batch['image']
            landmarks = batch['landmarks']

            input_imgs = imgs.to(self.device)
            ground_truth = landmarks.to(self.device)

            # Clears the gradients of all optimized
            self.optimizer.zero_grad()

            output = self.model(input_imgs)

            loss = self.criterion(output, ground_truth)

            # bp
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                self.log(
                    'Train Epoch: {} [{}/{} ({}/{})]\t loss: {:.6f}'.format(
                        epoch_title,
                        batch_idx *
                        len(imgs),
                        num_dataset,
                        batch_idx,
                        num_batch,
                        loss.item()
                    )
                )
        running_loss /= num_batch * 1.0
        return running_loss

    def do_validate(self, data_loader=None):
        data_loader = self.valid_data_loader if data_loader is None else data_loader

        num_batch = len(data_loader)

        # Sets the module in evaluation mode.
        self.model.eval()

        mean_loss = 0.0
        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                imgs = batch['image']
                landmarks = batch['landmarks']

                input_imgs = imgs.to(self.device)
                ground_truth = landmarks.to(self.device)

                output = self.model(input_imgs)

                loss = self.criterion(output, ground_truth)

                mean_loss += loss.item()

            mean_loss /= num_batch * 1.0
        return mean_loss

    def _train_phase(self):
        self.log('====> Start Training')
        args = self.args
        retry_id = self.next_retry_id
        self.next_retry_id = self.next_retry_id + 1

        train_losses = []
        valid_losses = []
        for idx in range(args.epochs):
            epoch_id = self.next_epoch_id
            self.next_epoch_id = self.next_epoch_id + 1
            epoch_title = '{}/{}'.format(idx, args.epochs)

            train_loss = self.do_train(epoch_title)
            train_losses.append(train_loss)

            valid_loss = self.do_validate()
            valid_losses.append(valid_loss)

            self.log('Train: loss: {:.6f}'.format(train_loss))
            self.log('Valid: loss: {:.6f}'.format(valid_loss))
            self.log('====================================================')

            self.sublog(
                train_loss=train_loss, valid_loss=valid_loss,
                tag='loss', epoch=epoch_id, retry_id=retry_id,
            )
            if np.isnan(train_loss) or np.isnan(valid_loss):
                if args.retry:
                    args.model_file = self.random_previous_model()
                    args.lr = args.lr / math.sqrt(3)
                    self.reset_model()

                    self.log(
                        'retry: lr: {}, model: {}'.format(args.lr, args.model_file))
                    self.sublog(
                        lr=args.lr, model_file=args.model_file,
                        tag='retry', epoch=epoch_id,
                    )
                return args.retry
            else:
                if args.save_model:
                    model_filepath = os.path.join(
                        self.save_subdir, 'epoch_{}.pt'.format(epoch_id))
                    torch.save(self.model.state_dict(),
                               model_filepath)
                self.draw_losses(train_losses, valid_losses)

        return False

    def train_phase(self):
        while self._train_phase():
            pass
        plt.show()

    def random_previous_model(self):
        import random
        models = filter(lambda x: x.startswith('epoch_'),
                        os.listdir(self.save_subdir))
        sorted_models = sorted(models, key=lambda x: int(x[6:-3]))

        offset = min(random.randrange(1, 6), len(sorted_models))
        if offset > 0:
            model_name = sorted_models[-offset]
            return os.path.join(self.save_subdir, model_name)
        return None

    def test_phase(self):
        self.log('====> Testing Model on the train set')
        train_loss = self.do_validate(self.train_data_loader)
        self.log('loss for the train set: {}'.format(train_loss))
        self.sublog(train_loss=train_loss)

        self.log('====> Testing Model on the test set')
        test_loss = self.do_validate()
        self.log('loss for the test set: {}'.format(test_loss))
        self.sublog(test_loss=test_loss)

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
    from util import parse_args, p

    args = parse_args('Detector', [
        p('--batch-size', type=int, default=64, metavar='N',
          help='input batch size for training (default: 64)'),
        p('--test-batch-size', type=int, default=64, metavar='N',
          help='input batch size for testing (default: 64)'),
        p('--epochs', type=int, default=100, metavar='N',
          help='number of epochs to train (default: 100)'),
        p('--lr', type=float, default=0.001, metavar='LR',
          help='learning rate (default: 0.00003)'),
        p('--momentum', type=float, default=0.5, metavar='M',
          help='SGD momentum (default: 0.5)'),
        p('--no-cuda', action='store_true', default=False,
          help='disables CUDA training'),
        p('--seed', type=int, default=1, metavar='S',
          help='random seed (default: 1)'),
        p('--log-interval', type=int, default=20, metavar='N',
          help='how many batches to wait before logging training status'),
        p('--save-model', action='store_true', default=True,
          help='save the current Model'),

        p('--save-directory', type=str, default='out',
          help='learnt models are saving here'),
        p('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
          help='training, predicting or finetuning'),
        p('--data-directory', type=str,
          help='data are loading from here'),

        p('--model-file', type=str,
          help='model are loading from here'),

        p('--predict-indices', type=str, default='all',
          help='sample indices to predict'),

        p('--no-cache-image', action='store_true', default=False,
            help='should cache image in memory'),

        p('--retry', action='store_true', default=True,
            help='loss为nan时是否自动需要调整参数重试'),

    ])

    return args


# %%
if __name__ == "__main__":
    args = _parse_args()

    # 随机种子
    torch.manual_seed(args.seed)

    detector = Detector(args)
    detector.run()
