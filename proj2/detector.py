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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        avgPool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv1_1 = nn.Conv2d(1, 8, kernel_size=5, stride=2)
        self.prelu1_1 = nn.PReLU()
        self.pool1 = avgPool

        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3)
        self.prelu2_2 = nn.PReLU()
        self.pool2 = avgPool

        self.conv3_1 = nn.Conv2d(16, 24, kernel_size=3)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(24, 24, kernel_size=3)
        self.prelu3_2 = nn.PReLU()
        self.pool3 = avgPool

        self.conv4_1 = nn.Conv2d(24, 40, kernel_size=3, padding=1)
        self.prelu4_1 = nn.PReLU()
        self.conv4_2 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.prelu4_2 = nn.PReLU()

        self.ip1 = nn.Linear(80 * 4 * 4, 128)
        self.preluip1 = nn.PReLU()
        self.ip2 = nn.Linear(128, 128)
        self.preluip2 = nn.PReLU()
        self.ip3 = nn.Linear(128, 42)

    def forward(self, X):
        """
        X: (1,1,112,112)
        retVal: (1, 42)
        """
        X = self.prelu1_1(self.conv1_1(X))
        X = self.pool1(X)

        X = self.prelu2_1(self.conv2_1(X))
        X = self.prelu2_2(self.conv2_2(X))
        X = self.pool2(X)

        X = self.prelu3_1(self.conv3_1(X))
        X = self.prelu3_2(self.conv3_2(X))
        X = self.pool3(X)

        X = self.prelu4_1(self.conv4_1(X))
        X = self.prelu4_2(self.conv4_2(X))

        X = X.view(-1, 80 * 4 * 4)
        X = self.preluip1(self.ip1(X))
        X = self.preluip2(self.ip2(X))
        X = self.ip3(X)

        return X


class Detector():

    def __init__(self, args):
        self._init_args(args)
        self._init_log(args)

        self._init_device(args)
        self.log("====> Loading Datasets")
        self._init_data_loader(args)
        self.log("====> Building Model")
        self._init_net_model(self.device)
        self._init_criterion(args)
        self._init_optimizer(args, self.model)

        self.load_state_dict()

    def _init_args(self, args):
        """
        参数预处理
        """
        self.args = args

        args.phase = args.phase.lower()

        def train(args):
            return os.path.join(args.save_directory,
                                '{}_{}_{}'.format(args.phase, args.epochs, args.lr).replace('.', '-'))

        def test(args):
            prefix = os.path.commonprefix([args.save_directory,
                                           args.model_file])
            model_file = args.model_file[len(prefix):].replace('/', '_')

            return os.path.join(args.save_directory, 'test_{}'.format(model_file))

        formatters = {
            'train': train,
            'test': test,
        }
        formatter = formatters.get(args.phase)
        if formatter:
            args.save_directory = formatter(args)

    def _init_log(self, args):
        import logging
        import sys

        logger = logging.getLogger('detector')
        logger.setLevel(logging.INFO)

        hd = logging.StreamHandler(sys.stdout)
        logger.addHandler(hd)

        if args.save_log:
            if not os.path.exists(args.save_directory):
                os.makedirs(args.save_directory)
            filename = os.path.join(args.save_directory, 'all.log')
            hd = logging.FileHandler(filename)
            formatter = logging.Formatter('[%(asctime)s]: %(message)s')
            hd.setFormatter(formatter)
            logger.addHandler(hd)

        self.log = logger.info

        self.log('args: {}\n'.format(args))

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
            args.data_directory)

        self.train_data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
        self.valid_data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.test_batch_size)

    def _init_net_model(self, device):
        """
        网络模型
        """
        self.model = Net().to(device)

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
        state_dict = torch.load(filepath)
        self.model.load_state_dict(state_dict)

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

        # mean_loss = 0.0
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

            # mean_loss += loss.item()
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
        # mean_loss /= num_batch * 1.0
        # return mean_loss

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

    def train_phase(self):
        self.log('====> Start Training')
        args = self.args

        if args.save_model:
            if not os.path.exists(args.save_directory):
                os.makedirs(args.save_directory)

        train_losses = []
        valid_losses = []
        for epoch_id in range(args.epochs):
            epoch_title = '{}/{}'.format(epoch_id, args.epochs)

            # train_loss = self.do_train(epoch_title)
            self.do_train(epoch_title)
            train_loss = self.do_validate(self.train_data_loader)
            train_losses.append(train_loss)

            valid_loss = self.do_validate()
            valid_losses.append(valid_loss)

            self.log('Train: loss: {:.6f}'.format(train_loss))
            self.log('Valid: loss: {:.6f}'.format(valid_loss))

            self.log('====================================================')

            if args.save_model:
                saved_model_name = os.path.join(
                    args.save_directory, 'detector_epoch_' + str(epoch_id) + '.pt')
                torch.save(self.model.state_dict(), saved_model_name)

            self.draw_losses(train_losses, valid_losses)

        if args.save_model:
            np.savetxt(os.path.join(
                args.save_directory, 'train_losses.txt'), train_losses)
            np.savetxt(os.path.join(
                args.save_directory, 'valid_losses.txt'), valid_losses)

    def test_phase(self):
        self.log('====> Testing Model on the train set')
        train_loss = self.do_validate(self.train_data_loader)
        self.log('loss for the train set: {}'.format(train_loss))

        self.log('====> Testing Model on the test set')
        test_loss = self.do_validate()
        self.log('loss for the test set: {}'.format(test_loss))

    def finetune_phase(self):
        self.train_phase()

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

                input_img = img.expand(1, *img.size())

                output = self.model(input_img)[0]

                plt.imshow(img[0])
                plt.scatter(landmarks[0::2], landmarks[1::2],
                            alpha=0.5, color='yellow')

                plt.scatter(output[0::2], output[1::2],
                            alpha=0.5, color='r')

                plt.show()


def _get_argparser():
    parser = argparse.ArgumentParser(description='Detector')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')

    parser.add_argument('--save-log', action='store_true', default=True,
                        help='save the logs')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    parser.add_argument('--data-directory', type=str,
                        help='data are loading from here')

    parser.add_argument('--model-file', type=str,
                        help='model are loading from here')

    parser.add_argument('--predict-indices', type=str, default='',
                        help='sample indices to predict')
    return parser


def _parse_args():
    args = _get_argparser().parse_args()
    return args


# %%
if __name__ == "__main__":
    args = _parse_args()

    # 随机种子
    torch.manual_seed(args.seed)

    detector = Detector(args)
    detector.run()

    plt.show()
