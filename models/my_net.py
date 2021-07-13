from torch import nn
from torch.nn import functional as F
import math

import torch as t
import torchvision.models as M



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x)



class Inception(nn.Module):

    def __init__(self, in_channels, branch1_1x1, branch1_3x3_1, branch1_3x3_2,
                 branch2_1x1, branch2_3x3, branch3_1x1, branch4_1x1):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, branch1_1x1, kernel_size=1, stride=1),
            BasicConv2d(branch1_1x1, branch1_3x3_1, kernel_size=3, stride=1, padding=1),
            BasicConv2d(branch1_3x3_1, branch1_3x3_2, kernel_size=3, stride=1, padding=1),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, branch2_1x1, kernel_size=1, stride=1),
            BasicConv2d(branch2_1x1, branch2_3x3, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True),
            BasicConv2d(in_channels, branch3_1x1, kernel_size=1, stride=1)
        )
        self.branch4 = BasicConv2d(in_channels, branch4_1x1, kernel_size=1, stride=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return t.cat([branch1, branch2, branch3, branch4], dim=1)



class SE(nn.Module):
    def __init__(self, in_channels):
        super(SE, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channels, in_channels//16, 1, 1, 0)
        self.excitation = nn.Conv2d(in_channels//16, in_channels, 1, 1, 0)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.compress(x)
        x = F.relu(x)
        x = self.excitation(x)
        return t.sigmoid(x)


class SPP(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(SPP, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out = None
        for n in self.out_side:
            w_r, h_r = map(lambda s: math.ceil(s / n), x.size()[2:])  # Receptive Field Size
            s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
            max_pool = nn.MaxPool2d(kernel_size=(w_r, h_r), stride=(s_w, s_h))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = t.cat((out, y.view(y.size()[0], -1)), 1)
        return out


class My_Net(nn.Module):
    def __init__(self, attribute_num):
        super(My_Net, self).__init__()

        # 前面初始卷积
        self.pre_layers = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            BasicConv2d(64, 64, kernel_size=1, stride=1),

            BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

        )

        # inception层
        self.inception1 = Inception(192, 64, 128, 128, 32, 128, 64, 128)
        self.inception2 = Inception(448, 128, 256, 256, 64, 256, 128, 256)

        self.inception3 = Inception(896, 64, 128, 128, 32, 128, 64, 128)

        # seNet 通道注意力
        self.se1 = SE(448)
        self.se2 = SE(896)
        self.se3 = SE(448)

        # 多特征图卷积融合
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels=448, out_channels=64, kernel_size=3, stride=1, padding=0),
            SPP(out_side=(1, 2, 4)),
            nn.Linear(64*(16+4+1), 512),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels=896, out_channels=64, kernel_size=3, stride=1, padding=0),
            SPP(out_side=(1, 2, 4)),
            nn.Linear(64 * (16 + 4 + 1), 512),
            nn.ReLU(inplace=True),

        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels=448, out_channels=64, kernel_size=3, stride=1, padding=0),
            SPP(out_side=(1, 2, 4)),
            nn.Linear(64 * (16 + 4 + 1), 512),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.ModuleList()
        for _ in range(attribute_num):
            self.classifier.append(nn.Sequential(
            nn.Linear(512*3, 512, bias=True),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        ))



    def forward(self, x):
        x = self.pre_layers(x)

        x = self.inception1(x)
        w1 = self.se1(x)

        x = x*w1

        ## branch1
        feature1 = self.branch1(x)

        x = self.inception2(x)
        w2 = self.se2(x)
        x = x*w2

        ## branch2
        feature2 = self.branch2(x)

        x = self.inception3(x)
        w3 = self.se3(x)
        x = x*w3

        ## brach3——Main Batch
        feature3 = self.branch3(x)

        # 特征融合
        x = t.cat([feature1, feature2, feature3], dim=1)

        total_out = []
        for classifier in self.classifier:
            out = classifier(x)
            total_out.append(out)

        # 合并特征
        outputs = t.cat(total_out, dim=1)
        return outputs







