import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 3*500*500 -> 3*498*498
        self.conv1 = nn.Conv2d(3, 3, 3)
        # 3*498*498 -> 3*249*249
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)

        # 3*249*249 -> 6*247*247
        self.conv2 = nn.Conv2d(3, 6, 3)
        # 6*247*247 -> 6*123*123
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU(inplace=True)

        # 6*123*123 -> 150
        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.relufc1 = nn.ReLU(inplace=True)

        self.dropfc1 = nn.Dropout2d()

        # 150 -> 2
        self.fc2 = nn.Linear(150, 2)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.relufc1(x)

        x = F.dropout(x, training=self.training)

        x_classes = self.fc2(x)
        x_classes = self.softmax2(x_classes)

        return x_classes
