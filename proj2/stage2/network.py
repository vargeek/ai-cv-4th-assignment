import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        # for p in self.resnet18.parameters():
        #     p.requires_grad = False

        self.ip1 = nn.Linear(1000, 128)
        self.preluip1 = nn.PReLU()
        self.ip2 = nn.Linear(128, 128)
        self.preluip2 = nn.PReLU()
        self.bn_ip3 = nn.BatchNorm1d(128)
        self.ip3 = nn.Linear(128, 42)

    def forward(self, x):
        """
        x: (1,1,112,112)
        retVal: (1, 42)
        """
        x = self.resnet18(x)

        x = self.preluip1(self.ip1(x))

        x = self.preluip2(self.ip2(x))

        x = self.bn_ip3(x)
        x = self.ip3(x)

        return x

    def init_parameters_kaiming(self):
        for module in self.children():
            if module != self.resnet18:
                for p in module.parameters():
                    if len(p.shape) >= 2:
                        nn.init.kaiming_normal_(p)
