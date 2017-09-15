import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable

from .resnet import Block, Bottleneck, _possible_downsample

import visualize


def residual(x, w1, b1, rm1, rv1, bw1, bb1, w2, b2, rm2, rv2, bw2, bb2,
        training, stride=1):
    """ Basic residual block in functional form
    
    Args:
    """
    out = F.conv2d(x, w1, b1, stride, padding=1)
    out = F.batch_norm(out, rm1, rv1, bw1, bb1, training)
    out = F.relu(out)

    out = F.conv2d(out, w2, b2, stride=1, padding=1)
    out = F.batch_norm(out, rm2, rv2, bw2, bb2, training)
    out += possible_downsample(x)
    out = F.relu(out)

    return out


class RevBlockFunction(Function):
    activations = []

    @staticmethod
    def forward(ctx, x, in_channels, out_channels, *res_args, stride=1,
                bottleneck=False):

        # if stride > 1 information is lost and we need to save the input
        if stride > 1:
            __class__.activations.append(x)
            ctx.load_input = True
        else:
            ctx.load_input = False

        x1, x2 = x.chunk(2, dim=1)

        x1_ = possible_downsample(x1, in_channels, out_channels, stride)
        x2_ = possible_downsample(x2, in_channels, out_channels, stride)

        f_x2 = residual(x2, *res_args[:12], training, stride=stride)

        y1 = f_x2 + x1_

        g_y1 = residual(y1, *res_args[12:], training)

        y2 = g_y1 + x2_

        y = torch.cat([y1.data, y2.data], dim=1)

        ctx.save_for_backward(*res_args)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        res_args = ctx.saved_variables

        dy1, dy2 = Variable.chunk(grad_out, 2, dim=1)

        def f(x):
            return residual(x, *res_args[:12], training)

        def g(x):
            return residual(x, *res_args[12:], training)

        if ctx.load_input:
            __class__.activations.pop()
            input = __class__.activations.pop()
            x1, x2 = input.chunk(2, dim=1)
        else:
            output = __class__.activations.pop()
            y1, y2 = Variable.chunk(output, 2, dim=1)
            x2 = y2 - g(y1)
            x1 = y1 - f(x2)
        
        x1.detach_()
        x2.detach_()

        x1, x2 = Variable(x1.data, requires_grad=True), Variable(x2.data, requires_grad=True)

        x1_ = _possible_downsample(x1, in_channels, out_channels, stride)
        x2_ = _possible_downsample(x2, in_channels, out_channels, stride)

        f_x2 = f(x2)

        y1_ = f_x2 + x1_

        g_y1 = g(y1_)

        y2_ = g_y1 + x2_

        torch.autograd.backward(y2_, dy2, retain_graph=True)
        dd1 = torch.autograd.grad(y2_, [y1_] + res_args[12:-4], dy2, retain_graph=True)[0]
        dy2_y1 = dd1[0]
        dgw = dd1[1:]
        dy1_plus = dy2_y1 + dy1
        torch.autograd.backward(y1_, dy1_plus, retain_graph=True)
        dd2 = torch.autograd.grad(y1_, [x1, x2] + res_args[:8], dy1_plus, retain_graph=True)
        dx2 = dd2[1]
        dx2 += torch.autograd.grad(x2_, x2, dy2, retain_graph=True)[0]
        dx1 = dd2[0]
        dfw = dd2[2:]

        x = torch.cat((x1, x2), 1)
        __class__.activations.append(x)

        dx = torch.cat((dx1, dx2), 1)


        return ((dx, None, None) + tuple(dfw) + tuple([None]*4) + tuple(dgw) +
                tuple([None]*4))


class RevBlock(nn.Module):
    function = RevBlockFunction
    def __init__(self, in_channels, out_channels, stride=1):
        super(RevBlock, self).__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.stride = stride

        w1 = nn.Parameter(torch.Tensor(self.out_channels,
                                       self.in_channels, 3, 3)
        b1 = nn.Parameter(torch.Tensor(self.out_channels))
        # TODO: rm1, rv1, bw1, bb1, w2, b2, rm2, rv2, bw2, bb2

    def forward(self, x):
        return function.apply(x, self.in_channels, self.out_channels,
                              self.stride)


def revblock_metaclass(registry):
    """Metaclass function to inject instance level activation registry

    Used to pass recreated block inputs as outputs to the previous layer.
    """

    patch = RevBlockFunction.__dict__.copy()
    patch["activations"] = registry
    function = type("RevBlockFunction", (Function,), patch)

    mod_patch = RevBlock.__dict__.copy()
    mod_patch["function"] = function

    return type("RevBlock", (nn.Module,), mod_patch)



class RevBottleneck(nn.Module):
    # TODO: Implement metaclass and function
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
        super(RevNet, self).__init__()
        self.name = self.__class__.__name__

        self.activations = []

        if bottleneck:
            self.Reversible = RevBottleneck     # TODO: Implement RevBottleneck
        else:
            self.Reversible = revblock_metaclass(self.activations)

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Conv2d(3, filters[0], 3, padding=1))
        self.layers.append(nn.BatchNorm2d(filters[0]))  # remove parameters?

        for i, group_i in enumerate(units):
            layers.append(self.Reversible(filters[i], filters[i + 1],
                                         stride=strides[i]))

            for unit in range(1, group_i):
                layers.append(self.Reversible(filters[i + 1],
                                             filters[i + 1]))


        self.fc = nn.Linear(filters[-1], classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
