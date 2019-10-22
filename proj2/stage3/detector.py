#!/usr/bin/env python
import init
from common import executor
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler


class Detector(executor.Executor):

    def _import_network(self):
        import network
        return network

    def _init_criterion(self):
        """
        损失函数
        """
        self.pred_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss()

    def _init_tracer(self):
        super(Detector, self)._init_tracer()
        from common import util
        from common.util import title_format_for_fields

        format1 = title_format_for_fields(['valid_loss'])
        format2 = title_format_for_fields(['val_acc'], reduce=max)
        format3 = title_format_for_fields(['pred_val_loss'])
        format4 = title_format_for_fields(['cls_val_loss'])
        formats = [[format1, format2], [format3, format4]]

        def title_format(metrics, row, col):
            format = formats[row][col]
            return format(metrics, row, col)

        self.tracer.title_format = title_format

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

    def get_dataset_class(self):
        import dataset
        return dataset.Stage3Dataset

    def _init_data_loader(self):
        """
         加载数据集: 测试集、验证集
         """
        import dataset

        args = self.args
        Dataset = self.get_dataset_class()

        train_set, test_set = Dataset.get_train_test_set(
            args)

        train_sampler = Dataset.get_sampler(
            args, 'train', args.batch_size, shuffle=True)

        test_sampler = Dataset.get_sampler(
            args, 'test', args.test_batch_size)

        self.train_data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
        self.valid_data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.test_batch_size, sampler=test_sampler, drop_last=True)

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

    def run_an_epoch(self, is_training=True, data_loader=None):
        args = self.args
        tracer = self.tracer

        if data_loader is None:
            data_loader = self.train_data_loader if is_training else self.valid_data_loader
        total_batch = len(data_loader)
        # total_samples = len(data_loader.dataset)
        total_samples = data_loader.batch_size * total_batch  # drop_last=True

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_pred_loss = 0.0
        running_cls_loss = 0.0
        running_acc = 0.0
        running_recall = 0.0
        running_precision = 0.0
        num_samples = 0
        num_corrected = 0

        running_tp = 0
        running_tn = 0
        running_fp = 0
        running_fn = 0

        for batch_idx, batch in enumerate(data_loader):
            inputs = batch['image'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            positive = batch['positive'].to(self.device)

            if is_training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):

                pred_out, cls_out = self.model(inputs)
                _, cls_idx = torch.max(cls_out, 1)

                pred_loss = self.pred_criterion(
                    pred_out[positive, :], landmarks[positive, :])
                cls_loss = self.cls_criterion(cls_out, positive)

                loss = pred_loss * args.pred_weight + cls_loss * args.cls_weight

                if is_training:
                    loss.backward()
                    self.optimizer.step()

            num_batch_samples = len(inputs)
            num_samples += num_batch_samples
            loss_value = loss.item()
            pred_loss_value = pred_loss.item()
            cls_loss_value = cls_loss.item()
            running_loss += loss_value
            running_pred_loss += pred_loss_value
            running_cls_loss += cls_loss_value

            positive_u8 = positive.to(torch.uint8)
            negative_u8 = 1 - positive_u8
            true_samples = (cls_idx == positive).to(torch.uint8)
            false_samples = (cls_idx != positive).to(torch.uint8)

            tp = torch.sum(true_samples + positive_u8 == 2).item()
            tn = torch.sum(true_samples + negative_u8 == 2).item()
            fp = torch.sum(false_samples + positive_u8 == 2).item()
            fn = num_batch_samples - tp - tn - fp

            running_tp += tp
            running_tn += tn
            running_fp += fp
            running_fn += fn

            corrected = tp + tn
            # corrected = torch.sum(cls_idx == positive).item()

            num_corrected += corrected

            tracer.batch_report(
                num_samples, total_samples,
                batch_idx+1, total_batch,
                loss=loss_value,
                corrected=num_corrected/num_samples,
            )

        running_loss /= total_batch * 1.0
        running_pred_loss /= total_batch * 1.0
        running_cls_loss /= total_batch * 1.0
        running_acc = num_corrected * 1.0 / total_samples
        running_recall = np.nan if (running_tp + running_fn) == 0 else running_tp / \
            (running_tp + running_fn)
        running_precision = np.nan if (running_tp + running_fn) == 0 else running_tp / \
            (running_tp + running_fp)

        return {
            'loss': running_loss,
            'pred_loss': running_pred_loss,
            'cls_loss': running_cls_loss,
            'acc': running_acc,
            'recall': running_recall,
            'precision': running_precision,
        }

    @executor.retry
    def _train_phase_(self):
        self.log('====> Start Training')
        args = self.args
        tracer = self.tracer

        tracer.epoch_reset()

        for idx in range(args.epochs):
            tracer.epoch_step(idx)

            train_metrics = self.run_an_epoch(True)
            val_metrics = self.run_an_epoch(False)
            train_loss = train_metrics['loss']
            pred_train_loss = train_metrics['pred_loss']
            cls_train_loss = train_metrics['cls_loss']

            valid_loss = val_metrics['loss']
            val_acc = val_metrics['acc']
            val_recall = val_metrics['recall']
            val_prec = val_metrics['precision']
            pred_val_loss = val_metrics['pred_loss']
            cls_val_loss = val_metrics['cls_loss']

            if np.isnan(train_loss) or np.isnan(valid_loss):
                return True

            grid = np.array([
                [
                    ['train_loss', 'valid_loss'],
                    ['val_acc', 'val_recall', 'val_prec'],
                ],
                [
                    ['pred_train_loss', 'pred_val_loss'],
                    ['cls_train_loss', 'cls_val_loss'],
                ],
            ])
            tracer.epoch_report([
                ('train_loss', train_loss),
                ('pred_train_loss', pred_train_loss),
                ('cls_train_loss', cls_train_loss),
                ('valid_loss', valid_loss),
                ('pred_val_loss', pred_val_loss),
                ('cls_val_loss', cls_val_loss),
                ('val_acc', val_acc),
                ('val_recall', val_recall),
                ('val_prec', val_prec),
            ], grid, retry_id=self.retry_id)

        return False


if __name__ == "__main__":
    # 随机种子

    parser, arg = Detector.get_args_parser('detector')
    arg('--cls-weight', type=int, default=3.0,
        help='weight of cls loss')

    arg('--pred-weight', type=int, default=1.0,
        help='weight of pred loss')

    args = parser.parse_args([] if
                             init.is_ipython_mode() else None)
    args.stage = init.STAGE

    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)

    exec = Detector(args)
    exec.run()
