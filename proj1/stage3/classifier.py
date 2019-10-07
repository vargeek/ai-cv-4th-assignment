#!/usr/bin/env python
# %%
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from loss_tracer import LossTracer

IPYTHON_MODE = 'get_ipython' in dir()
CUR_DIR = os.path.curdir if IPYTHON_MODE else os.path.dirname(
    __file__)
sys.path.append(os.path.join(CUR_DIR, '..'))

CLASSES = ['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']


def importExecutor():
    from utils import Executor
    return Executor


class Classifier(importExecutor()):
    def __init__(self, args):
        super(Classifier, self).__init__(args)

    def _import_dataset(self):
        import dataset
        return dataset

    def _init_tracer(self, args):
        self.tracer = LossTracer(args, self)

    def _init_net_model(self, args):
        """
        网络模型
        """
        import Multi_Network

        name = args.model
        if name != 'Net':
            name = 'Net_{}'.format(name)

        Net = getattr(Multi_Network, name) if hasattr(Multi_Network, name) else Multi_Network.Net

        self.model = Net().to(self.device)
        if self.args.model_file is None:
            self.init_parameters_kaiming()
        else:
            self.load_state_dict()

    def _init_criterion(self, args):
        """
        损失函数
        """
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
        num_corrected_s = 0
        num_corrected_c = 0
        num_samples = 0

        for batch_idx, batch in enumerate(data_loader):
            inputs = batch['image'].to(self.device)
            ground_truth_s = batch['species'].to(self.device)
            ground_truth_c = batch['classes'].to(self.device)

            if is_training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):
                outputs_species, outputs_classes = self.model(inputs)

                _, predicted_species = torch.max(outputs_species, 1)
                _, predicted_classes = torch.max(outputs_classes, 1)

                loss_species = self.criterion(outputs_species, ground_truth_s)
                loss_classes = self.criterion(outputs_classes, ground_truth_c)
                # loss = loss_species
                loss = loss_species * 0.9 + loss_classes * 0.1

                if is_training:
                    loss.backward()
                    self.optimizer.step()

            num_batch_samples = len(inputs)
            num_samples += num_batch_samples
            loss_value = loss.item()
            running_loss += loss_value

            corrected_s = torch.sum(predicted_species == ground_truth_s).item()
            corrected_c = torch.sum(predicted_classes == ground_truth_c).item()
            num_corrected_s += corrected_s
            num_corrected_c += corrected_c

            tracer.report_batch_loss(
                num_batch_samples, num_samples, total_samples, batch_idx + 1, total_batch, loss_value)

        running_loss /= total_batch * 1.0
        running_acc_s = num_corrected_s * 1.0 / total_samples
        running_acc_c = num_corrected_c * 1.0 / total_samples
        return running_loss, running_acc_s, running_acc_c

    # phases
    def _train_phase_(self):
        self.log('====> Start Training')
        tracer = self.tracer

        tracer.epoch_reset()
        for idx in range(args.epochs):
            tracer.epoch_step(idx)

            train_loss, train_acc_s, train_acc_c = self.train_or_validate_an_epoch(True)
            valid_loss, valid_acc_s, valid_acc_c = self.train_or_validate_an_epoch(False)
            val_acc = (valid_acc_s+valid_acc_c) if (valid_acc_s == 0 or valid_acc_c == 0) else 2.0/(1.0/valid_acc_s + 1.0/valid_acc_c)

            tracer.epoch_loss_report({
                'train_loss': train_loss, 
                'valid_loss': valid_loss, 
                },{
                'train_acc_s': train_acc_s, 
                'train_acc_c': train_acc_c,
                'val_acc_s': valid_acc_s,
                'val_acc_c': valid_acc_c,
                'val_acc': val_acc,
                })

    def _test_phase_(self):
        print('xxx')

    def _predict_phase_(self):
        dataset, indices = self.get_predict_dataset()
        self.model.eval()
        with torch.no_grad():
            for idx in indices:
                if idx < 0 or idx >= len(dataset):
                    continue
                sample = dataset[idx]

                inputs = sample['image']
                inputs = inputs.expand(1, *inputs.size()).to(self.device)

                ground_truth = sample['species']

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
                plt.title('predicted species: {}\nground-truth species: {}'.format(
                    SPECIES[predicted.item()], SPECIES[ground_truth]))
                plt.show()

def _get_args(args=None):
    parser, _ = Classifier.get_args_parser('Classifier')
    return parser.parse_args(args)


if __name__ == "__main__":
    args = _get_args([] if IPYTHON_MODE else None)

    # 随机种子
    torch.manual_seed(args.seed)

    classifier = Classifier(args)
    classifier.run()

