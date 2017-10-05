import torch
import torch.autograd
from torch.autograd import Variable

# import visualize

import models.revnet as revnet

import unittest

from .common import TestCase


class TestRevNet(TestCase):
    def setUp(self):
        pass

    def test_grad(self):
        act = []
        net = revnet.RevBlock(2, 2, act)
        x = Variable(torch.rand(1, 2, 4, 4), requires_grad=True)
        act.append(net(x).data)
        inputs = [
            x,
            net.in_channels,
            net.out_channels,
            net.training,
            net.stride,
            net.no_activation,
            net.activations,
            *net._parameters.values(),
            *net._buffers.values()
        ]
        test = torch.autograd.gradcheck(revnet.RevBlockFunction.apply, inputs)
        print(test)

    def test_recreation(self):
        pass


if __name__ == '__main__':
    unittest.main()
