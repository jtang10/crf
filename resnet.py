from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride, padding, bias=True):
        super(Conv1D_BN_ReLU, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, k_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self.conv3_1 = Conv1D_BN_ReLU(in_channels, 64, 3, 1, 1)
        self.conv3_2 = Conv1D_BN_ReLU(64, 128, 3, 1, 1)
        self.conv1 = Conv1D_BN_ReLU(128, 128, 1, 1, 0)
        self.conv3_3 = Conv1D_BN_ReLU(128, out_channels, 3, 1, 1)

    def forward(self, x):
        output = self.conv3_1(x)
        output = self.conv3_2(output)
        output = self.conv1(output)
        output = self.conv3_3(output)
        return output


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.conv1s = nn.ModuleList(
            [Conv1D_BN_ReLU(in_channels, 256, 1, 1, 0) for i in range(3)])

    def forward(self, x):
        output = x
        return output


class Inception_A(nn.Module):
    def __init__(self, in_channels, scale=0.1):
        super(Inception_A, self).__init__()
        self.scale = scale
        self.conv1s = nn.ModuleList(
            [Conv1D_BN_ReLU(in_channels, 32, 1, 1, 0) for i in range(3)])
        self.conv3_1 = Conv1D_BN_ReLU(32, 32, 3, 1, 1)
        self.conv3_2 = Conv1D_BN_ReLU(32, 48, 3, 1, 1)
        self.conv3_3 = Conv1D_BN_ReLU(48, 64, 3, 1, 1)
        self.conv1_cat = Conv1D_BN_ReLU(128, 384, 1, 1, 0)

    def forward(self, x):
        branch1 = self.conv1s[0](x)
        branch2 = self.conv3_1(self.conv1s[1](x))
        branch3 = self.conv3_3(self.conv3_2(self.conv1s[2](x)))
        output = torch.cat([branch1, branch2, branch3], 1)
        output = self.conv1_cat(output) * self.scale + x
        return output


class Inception_B(nn.Module):
    def __init__(self, in_channels, out_channels, scale=0.1):
        super(Inception_A, self).__init__()
        self.scale = scale
        self.branch1 = Conv1D_BN_ReLU(in_channels, 192, 1, 1, 0)
        self.branch2 = nn.Sequential(Conv1D_BN_ReLU(in_channels, 128, 1, 1, 0),
                                     Conv1D_BN_ReLU(128, 192, 5, 1, 2))
        self.branch3 = nn.Sequential(Conv1D_BN_ReLU(in_channels, 128, 1, 1, 0),
                                     Conv1D_BN_ReLU(128, 192, 7, 1, 3))
        self.conv1_cat = Conv1D_BN_ReLU(576, 1154, 1, 1, 0)
        self.conv1_x = Conv1D_BN_ReLU(in_channels, 1154, 1, 1, 0)

    def forward(self, x):
        residual = torch.cat(self.branch1(x), self.branch2(x), self.branch3(x))
        output = self.conv1_cat(residual) * self.scale + self.conv1_x(x)
        return output


class Inception_ResNet(nn.Module):
    def __init__(self, output_size=8):
        super(Inception_ResNet, self).__init__()
        self.stem = Stem(66, 384)
        self.inceptions = nn.ModuleList([Inception_A(384) for i in range(4)])
        self.linear = nn.Linear(384, output_size)

    def forward(self, x):
        output = self.stem(x)
        for i, inception in enumerate(self.inceptions):
            output = inception(output)
        output = self.linear(output.permute(0, 2, 1))
        return output


def get_n_params(model):
        pp=0
        for name, p in model.named_parameters():
            # print(name)
            # print(p.size())
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

if __name__ == '__main__':
    img = torch.autograd.Variable(torch.randn(8, 66, 698))
    model = Inception_ResNet()
    # stem = Stem(66, 384)
    # inception = Inception_A(384, 384)
    print("Number of trainable parameters", get_n_params(model))
    output = model(img)
    print(output.size())