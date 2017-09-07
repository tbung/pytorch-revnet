import torch
import torch.nn as nn

from torch.autograd import Function

from .resnet import Block


# class RevBlockFunction(Function):
#     @staticmethod
#     def forward(ctx, x,


class RevBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RevBlock, self).__init__()

        self.f = Block(in_channels, out_channels, stride)
        self.g = Block(out_channels, out_channels, stride=1)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=3)

        f_x2 = self.f(x2)

        y1 = f_x2 + x1

        g_y1 = self.g(y1)

        y2 = g_y1 + x2

        return torch.cat([y1, y2], dim=3)


class RevBottleneck(nn.Module):
    pass


class RevNet(nn.Module):
    pass
