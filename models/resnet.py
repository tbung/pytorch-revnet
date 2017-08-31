import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()

        self.stride = stride


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=1, stride=stride)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)


        if stride != 1:
            self.res = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                    )

    def forward(self, x):
        orig_x = x

        out = F.relu(self.bn1(self.conv1(x)))

        if self.res is not None:
            orig_x = self.res(orig_x)

        out = F.relu(self.bn2(self.conv2(x)) + orig_x)

        return out


class Bottleneck(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class RevNet(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
