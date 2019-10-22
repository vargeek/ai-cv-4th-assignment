#!/usr/bin/env python
import init
from common.dataset import FaceLandmarksDataset
from common import dataset as base
import numpy as np
import random
from common import util

from torch.utils.data import Sampler, BatchSampler


class Stage3TrainSampler(Sampler):
    def __init__(self, data_source, batch_sizes, grouping=None, shuffle=False):
        super(Stage3TrainSampler, self).__init__(data_source)
        assert(len(batch_sizes) > 0)

        self.data_source = data_source
        self.pos_batch_size, self.neg_batch_size = batch_sizes
        self.shuffle = shuffle

        if grouping is None:
            def is_positive(x):
                return x
            grouping = is_positive

        self.grouping = grouping

    def __iter__(self):
        data_source = self.data_source
        grouping = self.grouping
        indices = list(range(len(data_source)))
        if self.shuffle:
            random.shuffle(indices)

        group = {i: grouping(data_source[i]) for i in indices}
        indices_pos = [i for i in indices if group[i]]
        indices_neg = [i for i in indices if not group[i]]

        pos_batch_size = self.pos_batch_size
        neg_batch_size = self.neg_batch_size

        batch_pos = util.group(pos_batch_size, indices_pos)
        batch_neg = util.group(neg_batch_size, indices_neg)

        for x, y in zip(batch_pos, batch_neg):
            yield from x
            yield from y

    def __len__(self):
        return len(self.data_source)


class Stage3Dataset(FaceLandmarksDataset):
    def get_sample(self, idx):
        sample = super(Stage3Dataset, self).get_sample(idx)
        if sample.get('positive') is None:
            landmarks = sample['landmarks']
            positive = len(landmarks) > 0
            sample['positive'] = 1 if positive else 0
            if not positive:
                sample['landmarks'] = np.zeros(42, dtype=np.float32)
        return sample

    @classmethod
    def parse_positive(self, line):
        _, _, landmarks = base.parse_line(line)
        return len(landmarks) > 0

    @classmethod
    def get_is_positive(self, args, phase):
        import os
        data_dir = args.data_directory
        stage = args.stage
        filename = os.path.join(stage, phase + '.txt')

        filepath = os.path.join(data_dir, filename)
        with open(filepath) as f:
            lines = f.readlines()
            is_positive = [self.parse_positive(line) for line in lines]

        return is_positive

    @classmethod
    def get_sampler(self, args, phase, batch_size, **kwargs):
        import os
        datasource = self.get_is_positive(args, phase)

        num_pos = len(list(filter(lambda x: x, datasource)))

        pos = round(num_pos / len(datasource) * batch_size)
        neg = batch_size - pos

        sampler = Stage3TrainSampler(
            datasource, (pos, neg), **kwargs)
        return sampler


        # %%
if __name__ == "__main__":
    args = Stage3Dataset.parse_args()
    args.stage = init.STAGE
    Stage3Dataset.run(args)
