import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable

from .resnet import Block, Bottleneck, _possible_downsample


class RevBlockFunction(Function):
    @staticmethod
    def forward(ctx, input, f, g, in_channels, out_channels, bottleneck=False):
        x1, x2 = input.chunk(2, dim=1)

        x1, x2 = Variable(x1), Variable(x2)

        f_x2 = f(x2)

        x1_ = _possible_downsample(x1, in_channels, out_channels)
        x2_ = _possible_downsample(x2, in_channels, out_channels)

        y1 = f_x2 + x1_

        g_y1 = g(y1)

        y2 = g_y1 + x2_

        y = torch.cat([y1.data, y2.data], dim=1)

        return y

    @staticmethod
    def backward(output, grad_out, f, g, in_channels, out_channels):
        y1, y2 = Variable.chunk(output, 2, dim=1)
        dy1, dy2 = Variable.chunk(grad_out, 2, dim=1)

        x2 = y2 - g(y1)

        x1 = y1 - f(x2)

        f_x2 = f(x2)

        x1_ = _possible_downsample(x1, in_channels, out_channels)
        x2_ = _possible_downsample(x2, in_channels, out_channels)

        y1_ = f_x2 + x1_

        g_y1 = g(y1_)

        y2_ = g_y1 + x2_

        print("test 1")

        dd1 = torch.autograd.grad(y2_, [y1_] + list(g.parameters()), dy2)
        dy2_y1 = dd1[0]
        dy1_plus = dy2_y1 + dy1
        dgw = dd1[1:]
        dd2 = torch.autograd.grad(y1_, [x1, x2] + list(f.parameters()),
                                  dy1_plus)
        dx1 = dd2[0]
        dx2 = dd2[1]
        dfw = dd2[2:]
        dx2 += torch.autograd.grad(x2_, x2, dy2)[0]

        x = torch.cat((x1, x2), 1)
        dx = torch.cat((dx1, dx2), 1)

        print("test 2")

        return x, dx, dfw, dgw


class RevBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RevBlock, self).__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2

        self.f = Block(self.in_channels, self.out_channels, stride)
        self.g = Block(self.out_channels, self.out_channels, stride=1)

    def forward(self, x):
        return RevBlockFunction.apply(x, self.f, self.g,
                                      self.in_channels, self.out_channels)


class RevBottleneck(nn.Module):
    pass


class RevGroupFunction(Function):
    @staticmethod
    def forward(ctx, input, modules):
        for module in modules:
            input = module(input)

        ctx.save_for_backward(input.data)
        ctx._modules = modules
        return input.data

    @staticmethod
    def backward(ctx, grad_out):
        output = ctx.saved_variables[0]

        grad_modules = []

        for module in reversed(ctx._modules):
            output, grad_out, grad_wf, grad_wg = RevBlockFunction.backward(
                                                         output, grad_out,
                                                         module._modules['f'],
                                                         module._modules['g'],
                                                         module.in_channels,
                                                         module.out_channels)
            grad_modules.append(grad_wf)
            grad_modules.append(grad_wg)

        return grad_out, grad_modules


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
        super(RevNet, self).__init__()
        self.name = self.__class__.__name__

        if bottleneck:
            self.Residual = Bottleneck
            self.Reversible = RevBottleneck
        else:
            self.Residual = Block
            self.Reversible = RevBlock

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Conv2d(3, filters[0], 3, padding=1))
        self.layers.append(nn.BatchNorm2d(filters[0]))  # remove parameters?

        self.groups = nn.ModuleList()

        for i, group_i in enumerate(units):
            # if strides[i] != 1:
            #     self.layers.append(self.Residual(filters[i], filters[i + 1],
            #                                      stride=strides[i]))
            #     j = 1

            group = nn.ModuleList()

            group.append(self.Reversible(filters[i], filters[i + 1]))

            for unit in range(1, group_i):
                group.append(self.Reversible(filters[i + 1],
                                             filters[i + 1]))

            self.groups.append(group)

        self.fc = nn.Linear(filters[-1], classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        for group in self.groups:
            x = RevGroupFunction.apply(x, group)

        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
