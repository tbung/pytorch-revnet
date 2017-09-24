import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable

import gc


def free():
    del activations[:]
    gc.collect()


CUDA = torch.cuda.is_available()


def possible_downsample(x, in_channels, out_channels, stride=1):
    out = None
    if stride > 1:
        out = F.avg_pool2d(x, stride, stride)

    if in_channels < out_channels:
        if out is None:
            out = x
        pad = Variable(torch.zeros(out.size(0),
                                   (out_channels - in_channels) // 2,
                                   out.size(2), out.size(3)))
        if CUDA:
            pad = pad.cuda()

        temp = torch.cat([pad, out], dim=1)
        out = torch.cat([temp, pad], dim=1)

    if out is None:
        injection = Variable(torch.zeros_like(x.data))
        if CUDA:
            injection.cuda()
        out = x + injection

    return out


def residual(x, w1, b1, bw1, bb1, w2, b2, bw2, bb2, rm1, rv1, rm2, rv2,
             training, stride=1, no_activation=False):
    """ Basic residual block in functional form

    Args:
    """
    out = x
    if not no_activation:
        out = F.batch_norm(out, rm1, rv1, bw1, bb1, training)
        out = F.relu(out)
    out = F.conv2d(out, w1, b1, stride, padding=1)

    out = F.batch_norm(out, rm2, rv2, bw2, bb2, training)
    out = F.relu(out)
    out = F.conv2d(out, w2, b2, stride=1, padding=1)
    out = out + possible_downsample(x, w1.size(1), w1.size(0), stride)

    return out


activations = []


class RevBlockFunction(Function):
    @staticmethod
    def _inner(x, in_channels, out_channels, training, stride,
               f_params, f_buffs, g_params, g_buffs, manual_grads=True,
               no_activation=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if manual_grads:
            x1 = Variable(x1, volatile=True).contiguous()
            x2 = Variable(x2, volatile=True).contiguous()

        if CUDA:
            x1.cuda()
            x2.cuda()

        x1_ = possible_downsample(x1, in_channels, out_channels, stride)
        x2_ = possible_downsample(x2, in_channels, out_channels, stride)

        f_x2 = residual(x2, *f_params, *f_buffs, training, stride=stride,
                        no_activation=no_activation)

        y1 = f_x2 + x1_

        g_y1 = residual(y1, *g_params, *g_buffs, training)

        y2 = g_y1 + x2_

        y = torch.cat([y1, y2], dim=1)

        del y1, y2
        del x1, x2

        return y

    @staticmethod
    def _inner_backward(output, f_params, f_buffs, g_params, g_buffs,
                        training, no_activation):

        y1, y2 = torch.chunk(output, 2, dim=1)
        y1 = Variable(y1, volatile=True).contiguous()
        y2 = Variable(y2, volatile=True).contiguous()
        x2 = y2 - residual(y1, *g_params, *g_buffs, training=training)
        x1 = y1 - residual(x2, *f_params, *f_buffs, training=training,
                           no_activation=no_activation)
        del y1, y2
        x1, x2 = x1.data, x2.data

        x = torch.cat((x1, x2), 1)
        return x

    @staticmethod
    def _inner_grad(x, dy, in_channels, out_channels, training,
                    stride, f_params, f_buffs, g_params, g_buffs,
                    no_activation=False):
        dy1, dy2 = Variable.chunk(dy, 2, dim=1)

        x1, x2 = torch.chunk(x, 2, dim=1)

        x1 = Variable(x1, requires_grad=True).contiguous()
        x2 = Variable(x2, requires_grad=True).contiguous()

        if CUDA:
            x1.cuda()
            x2.cuda()

        x1_ = possible_downsample(x1, in_channels, out_channels, stride)
        x2_ = possible_downsample(x2, in_channels, out_channels, stride)

        f_x2 = residual(x2, *f_params, *f_buffs, training=training,
                        stride=stride, no_activation=no_activation)

        y1_ = f_x2 + x1_

        g_y1 = residual(y1_, *g_params, *g_buffs, training=training)
        y2_ = g_y1 + x2_

        dd1 = torch.autograd.grad(y2_, (y1_,) + tuple(g_params), dy2,
                                  retain_graph=True)
        dy2_y1 = dd1[0]
        dgw = dd1[1:]
        dy1_plus = dy2_y1 + dy1

        if not no_activation:
            dd2 = torch.autograd.grad(y1_, (x1, x2) + tuple(f_params),
                                      dy1_plus, retain_graph=True)
            dfw = dd2[2:]
        else:
            dd2 = torch.autograd.grad(y1_, (x1, x2) + tuple(f_params[:2]) +
                                      tuple(f_params[4:]),
                                      dy1_plus, retain_graph=True)
            dfw = dd2[2:4]
            dfw = dfw + (Variable(torch.zeros_like(f_params[2].data)),)
            dfw = dfw + (Variable(torch.zeros_like(f_params[3].data)),)
            dfw += dd2[4:]

        dx2 = dd2[1]
        dx2 += torch.autograd.grad(x2_, x2, dy2, retain_graph=True)[0]
        dx1 = dd2[0]

        activations.append(x)

        y1_.detach_()
        y2_.detach_()
        del y1_, y2_
        dx = torch.cat((dx1, dx2), 1)

        return dx, dfw, dgw

    @staticmethod
    def forward(ctx, x, in_channels, out_channels,
                training, stride, no_activation, *args):

        f_params = [Variable(p, volatile=True).contiguous() for p in args[:8]]
        f_buffs = args[8:12]
        g_params = [Variable(p, volatile=True).contiguous() for p in args[12:20]]
        g_buffs = args[20:]

        # if stride > 1 information is lost and we need to save the input
        if stride > 1:
            activations.append(x)
            ctx.load_input = True
        else:
            ctx.load_input = False

        ctx.save_for_backward(*args[:8], *args[12:20])
        ctx.f_buffs = args[8:12]
        ctx.g_buffs = args[20:]
        ctx.stride = stride
        ctx.training = training
        ctx.no_activation = no_activation

        y = RevBlockFunction._inner(x, in_channels, out_channels, training,
                                    stride, f_params, f_buffs, g_params,
                                    g_buffs, no_activation=no_activation)

        return y.data

    @staticmethod
    def backward(ctx, grad_out):
        f_params = ctx.saved_variables[:8]
        f_buffs = ctx.f_buffs
        g_params = ctx.saved_variables[8:]
        g_buffs = ctx.g_buffs

        in_channels = f_params[0].size(1)
        out_channels = f_params[0].size(0)

        if ctx.load_input:
            activations.pop()
            x = activations.pop()
        else:
            output = activations.pop()
            x = RevBlockFunction._inner_backward(output, f_params,
                                                 f_buffs, g_params,
                                                 g_buffs, ctx.training,
                                                 ctx.no_activation)

        dx, dfw, dgw = RevBlockFunction._inner_grad(x, grad_out,
                                                    in_channels, out_channels,
                                                    ctx.training, ctx.stride,
                                                    f_params, f_buffs,
                                                    g_params, g_buffs,
                                                    no_activation=ctx.no_activation)

        return ((dx, None, None, None, None, None) + tuple(dfw) + tuple([None]*4) +
                tuple(dgw) + tuple([None]*4))


class RevBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 no_activation=False):
        super(self.__class__, self).__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.stride = stride
        self.no_activation = no_activation

        self.f_params = []
        self.g_params = []

        # Conv 1
        self.f_params.append(nn.Parameter(torch.Tensor(self.out_channels,
                                          self.in_channels, 3, 3)))
        self.f_params.append(nn.Parameter(torch.Tensor(self.out_channels)))

        # BN 1
        self.f_params.append(nn.Parameter(torch.Tensor(self.in_channels)))
        self.f_params.append(nn.Parameter(torch.Tensor(self.in_channels)))

        # Conv 2
        self.f_params.append(nn.Parameter(torch.Tensor(self.out_channels,
                                          self.out_channels, 3, 3)))
        self.f_params.append(nn.Parameter(torch.Tensor(self.out_channels)))

        # BN 2
        self.f_params.append(nn.Parameter(torch.Tensor(self.out_channels)))
        self.f_params.append(nn.Parameter(torch.Tensor(self.out_channels)))

        # Conv 1
        self.g_params.append(nn.Parameter(torch.Tensor(self.out_channels,
                                          self.out_channels, 3, 3)))
        self.g_params.append(nn.Parameter(torch.Tensor(self.out_channels)))

        # BN 1
        self.g_params.append(nn.Parameter(torch.Tensor(self.out_channels)))
        self.g_params.append(nn.Parameter(torch.Tensor(self.out_channels)))

        # Conv 2
        self.g_params.append(nn.Parameter(torch.Tensor(self.out_channels,
                                          self.out_channels, 3, 3)))
        self.g_params.append(nn.Parameter(torch.Tensor(self.out_channels)))

        # BN 2
        self.g_params.append(nn.Parameter(torch.Tensor(self.out_channels)))
        self.g_params.append(nn.Parameter(torch.Tensor(self.out_channels)))

        param_names = ['w1', 'b1', 'bw1', 'bb1', 'w2', 'b2', 'bw2', 'bb2']

        for i, p in enumerate(self.f_params):
            self.register_parameter('f_'+param_names[i], p)

        for i, p in enumerate(self.g_params):
            self.register_parameter('g_'+param_names[i], p)

        self.register_buffer('f_rm1', torch.zeros(self.out_channels))
        self.register_buffer('f_rv1', torch.ones(self.out_channels))

        self.register_buffer('f_rm2', torch.zeros(self.out_channels))
        self.register_buffer('f_rv2', torch.ones(self.out_channels))

        self.register_buffer('g_rm1', torch.zeros(self.out_channels))
        self.register_buffer('g_rv1', torch.ones(self.out_channels))

        self.register_buffer('g_rm2', torch.zeros(self.out_channels))
        self.register_buffer('g_rv2', torch.ones(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        f_stdv = 1 / math.sqrt(self.in_channels * 3 * 3)
        g_stdv = 1 / math.sqrt(self.out_channels * 3 * 3)

        self._parameters['f_w1'].data.uniform_(-f_stdv, f_stdv)
        self._parameters['f_b1'].data.uniform_(-f_stdv, f_stdv)
        self._parameters['f_w2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['f_b2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['f_bw1'].data.uniform_()
        self._parameters['f_bb1'].data.zero_()
        self._parameters['f_bw2'].data.uniform_()
        self._parameters['f_bb2'].data.zero_()

        self._parameters['g_w1'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_b1'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_w2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_b2'].data.uniform_(-g_stdv, g_stdv)
        self._parameters['g_bw1'].data.uniform_()
        self._parameters['g_bb1'].data.zero_()
        self._parameters['g_bw2'].data.uniform_()
        self._parameters['g_bb2'].data.zero_()

        self.f_rm1.zero_()
        self.f_rv1.fill_(1)
        self.f_rm2.zero_()
        self.f_rv2.fill_(1)

        self.g_rm1.zero_()
        self.g_rv1.fill_(1)
        self.g_rm2.zero_()
        self.g_rv2.fill_(1)

    def forward(self, x):
        return RevBlockFunction.apply(x, self.in_channels, self.out_channels,
                                      self.training, self.stride,
                                      self.no_activation,
                                      *self.f_params,
                                      *list(self._buffers.values())[:4],
                                      *self.g_params,
                                      *list(self._buffers.values())[4:])


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

        if bottleneck:
            self.Reversible = RevBottleneck     # TODO: Implement RevBottleneck
        else:
            self.Reversible = RevBlock

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Conv2d(3, filters[0], 3, padding=1))
        self.layers.append(nn.BatchNorm2d(filters[0]))

        for i, group_i in enumerate(units):
            self.layers.append(self.Reversible(filters[i], filters[i + 1],
                                               stride=strides[i],
                                               no_activation=True))

            for unit in range(1, group_i):
                self.layers.append(self.Reversible(filters[i + 1],
                                                   filters[i + 1]))

        self.fc = nn.Linear(filters[-1], classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        activations.append(x.data)

        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
