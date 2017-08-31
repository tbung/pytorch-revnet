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
        else:
            self.res = lambda x: x

    def forward(self, x):
        orig_x = x

        out = F.relu(self.bn1(self.conv1(x)))

        orig_x = self.res(orig_x)

        out = F.relu(self.bn2(self.conv2(x)) + orig_x)

        return out


class Bottleneck(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class RevNet(nn.Module):
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

        if bottleneck:
            self.Residual = Bottleneck
        else:
            self.Residual = Block

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Conv2d(3, filters[0], padding=1))
        self.layers.append(nn.BatchNorm2d(filters[0]))

        for i, group in enumerate(units):
            self.layers.append(self.Residual(filters[i - 1], filters[i],
                                             stride=strides[i]))

            for unit in range(1, group):
                self.layers.append(self.Residual(filters[i], filters[i]))

        self.fc = nn.Linear(filters[-1], classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = F.avg_pool2d(x, x.size(0))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
