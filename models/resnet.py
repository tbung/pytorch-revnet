import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .revnet import possible_downsample

CUDA = torch.cuda.is_available()


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 no_activation=False):
        super(Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.no_activation = no_activation

        self.stride = stride

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)

    def forward(self, x):
        orig_x = x

        out = x

        if not self.no_activation:
            out = F.relu(self.bn1(out))

        out = self.conv1(out)

        out = self.conv2(F.relu(self.bn2(out)))

        out += possible_downsample(orig_x, self.in_channels,
                                   self.out_channels, self.stride)

        return out


class Bottleneck(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class ResNet(nn.Module):
    def __init__(self,
                 units,
                 filters,
                 strides,
                 classes,
                 bottleneck=False):
        """
        Parameters
        ----------

        units: list-like
            Number of residual units in each group

        filters: list-like
            Number of filters in each unit including the inputlayer, so it is
            one item longer than units

        strides: list-like
            Strides to use for the first units in each group, same length as
            units

        bottleneck: boolean
            Wether to use the bottleneck residual or the basic residual
        """
        super(ResNet, self).__init__()
        self.name = self.__class__.__name__

        if bottleneck:
            self.Residual = Bottleneck
        else:
            self.Residual = Block

        self.layers = nn.ModuleList()

        # Input layers
        self.layers.append(nn.Conv2d(3, filters[0], 3, padding=1))
        self.layers.append(nn.BatchNorm2d(filters[0]))
        self.layers.append(nn.ReLU())

        for i, group in enumerate(units):
            self.layers.append(self.Residual(filters[i], filters[i + 1],
                                             stride=strides[i],
                                             no_activation=True))

            for unit in range(1, group):
                self.layers.append(self.Residual(filters[i + 1],
                                                 filters[i + 1]))

        self.bn_last = nn.BatchNorm2d(filters[-1])

        self.fc = nn.Linear(filters[-1], classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = F.relu(self.bn_last(x))
        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
