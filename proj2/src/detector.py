# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%


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


net = Net()
print(net)
input = torch.randn(1, 1, 112, 112)
out = net(input)
print(out)

# %%
