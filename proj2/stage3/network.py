import torch
import torch.nn as nn
import torch.nn.functional as F


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

        ####### prediction branch #######
        # 40*4*4 -> 80*4*4
        self.conv4_2 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.prelu4_2 = nn.PReLU()

        self.ip1 = nn.Linear(80 * 4 * 4, 128)
        self.preluip1 = nn.PReLU()
        self.ip2 = nn.Linear(128, 128)
        self.preluip2 = nn.PReLU()
        self.landmarks = nn.Linear(128, 42)

        ####### classification branch #######
        # 40*4*4 -> 40*4*4
        self.conv4_2_cls = nn.Conv2d(40, 40, kernel_size=3, padding=1)
        self.relu_conv4_2_cls = nn.PReLU()

        self.ip1_cls = nn.Linear(40 * 4 * 4, 128)
        self.relu_ip1_cls = nn.PReLU()
        self.ip2_cls = nn.Linear(128, 128)
        self.relu_ip2_cls = nn.PReLU()
        self.ip3_cls = nn.Linear(128, 2)
        # self.face_score = nn.Softmax(dim=1)

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

        ####### prediction branch #######
        x_pred = self.prelu4_2(self.conv4_2(x))

        x_pred = x_pred.view(-1, 80 * 4 * 4)
        x_pred = self.preluip1(self.ip1(x_pred))

        x_pred = self.preluip2(self.ip2(x_pred))
        landmarks = self.landmarks(x_pred)

        ####### classification branch #######
        x_cls = self.relu_conv4_2_cls(self.conv4_2_cls(x))

        x_cls = x_cls.view(-1, 40 * 4 * 4)
        x_cls = self.relu_ip1_cls(self.ip1_cls(x_cls))
        x_cls = self.relu_ip2_cls(self.ip2_cls(x_cls))
        face = self.ip3_cls(x_cls)

        return landmarks, face
        # return face


class Net_BN(nn.Module):
    def __init__(self):
        super(Net_BN, self).__init__()

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

        ####### prediction branch #######
        # 40*4*4 -> 80*4*4
        self.conv4_2 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.prelu4_2 = nn.PReLU()

        self.ip1 = nn.Linear(80 * 4 * 4, 128)
        self.preluip1 = nn.PReLU()
        self.ip2 = nn.Linear(128, 128)
        self.preluip2 = nn.PReLU()
        self.landmarks_bn = nn.BatchNorm1d(128)
        self.landmarks = nn.Linear(128, 42)

        ####### classification branch #######
        # 40*4*4 -> 40*4*4
        self.conv4_2_cls = nn.Conv2d(40, 40, kernel_size=3, padding=1)
        self.relu_conv4_2_cls = nn.PReLU()

        self.ip1_cls = nn.Linear(40 * 4 * 4, 128)
        self.relu_ip1_cls = nn.PReLU()
        self.ip2_cls = nn.Linear(128, 128)
        self.relu_ip2_cls = nn.PReLU()
        self.ip3_cls = nn.Linear(128, 2)
        # self.face_score = nn.Softmax(dim=1)

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

        ####### prediction branch #######
        x_pred = self.prelu4_2(self.conv4_2(x))

        x_pred = x_pred.view(-1, 80 * 4 * 4)
        x_pred = self.preluip1(self.ip1(x_pred))

        x_pred = self.preluip2(self.ip2(x_pred))
        x_pred = self.landmarks_bn(x_pred)
        landmarks = self.landmarks(x_pred)

        ####### classification branch #######
        x_cls = self.relu_conv4_2_cls(self.conv4_2_cls(x))

        x_cls = x_cls.view(-1, 40 * 4 * 4)
        x_cls = self.relu_ip1_cls(self.ip1_cls(x_cls))
        x_cls = self.relu_ip2_cls(self.ip2_cls(x_cls))
        face = self.ip3_cls(x_cls)

        return landmarks, face
        # return face
