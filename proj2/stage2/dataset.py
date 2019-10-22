#!/usr/bin/env python
import matplotlib.pyplot as plt
import init
from common.dataset import FaceLandmarksDataset, parse_line
import os
import numpy as np
from PIL import Image
train_boarder = 112
# train_boarder = 224


class Stage2Dataset(FaceLandmarksDataset):
    def get_sample(self, idx):
        img_name, rect, landmarks = parse_line(self.lines[idx])
        img_path = os.path.join(self.data_dir, img_name)

        # img = Image.open(img_path).convert('L')
        img = Image.open(img_path).convert('RGB')
        img_crop = img.crop(tuple(rect))
        w, h = img_crop.size

        img_crop = img_crop.resize(
            (train_boarder, train_boarder), Image.BILINEAR)

        img_crop = np.array(img_crop, dtype=np.float32).transpose((2, 0, 1))

        landmarks = np.array(landmarks, dtype=np.float32)

        # you should let your landmarks fit to the train_boarder(112)
        # please complete your code under this blank
        # your code:
        x0, y0, *_ = rect
        landmarks[0::2] = (landmarks[0::2] - x0) * train_boarder / w
        landmarks[1::2] = (landmarks[1::2] - y0) * train_boarder / h

        sample = {
            'image': img_crop,
            'landmarks': landmarks,
            'image_name': img_name,
        }
        return sample

    @classmethod
    def show_dataset(self, dataset):
        for i in range(0, len(dataset)):
            sample = dataset[i]
            img = sample['image']
            landmarks = sample['landmarks']
            image_name = sample['image_name']
            # 请画出人脸crop以及对应的landmarks
            # please complete your code under this blank

            plt.title(image_name)
            plt.imshow(img[0, 0, :, :]+img[0, 1, :, :]+img[0, 2, :, :])

            plt.scatter(landmarks[0::2], landmarks[1::2],
                        alpha=0.5, color='r')
            plt.show()


# %%
if __name__ == "__main__":
    args = Stage2Dataset.parse_args()
    args.stage = init.STAGE

    Stage2Dataset.run(args)
