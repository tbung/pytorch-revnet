import torch
import torch.autograd
from torch.autograd import Variable

from revnet import RevBlock, RevBlockFunction

import unittest

from .common import TestCase


class TestRevNet(TestCase):
    def setUp(self):
        self.x = torch.rand(4, 4, 4, 4)
        self.model = RevBlock(4, 4, [])
        parameters = list(self.model._parameters.values())
        buffers = list(self.model._buffers.values())
        # self.f_params = [Variable(x) for x in parameters[:8]]
        # self.g_params = [Variable(x) for x in parameters[8:16]]
        self.f_params = parameters[:8]
        self.g_params = parameters[8:16]
        self.f_buffs = buffers[:4]
        self.g_buffs = buffers[4:8]
        self.in_channels = self.model.in_channels
        self.out_channels = self.model.out_channels
        self.training = self.model.training
        self.stride = self.model.stride
        self.no_activation = self.model.no_activation

    def test_grad(self):
        pass

    def test_recreation(self):
        y = RevBlockFunction._forward(
            self.x,
            self.in_channels,
            self.out_channels,
            self.training,
            self.stride,
            self.f_params, self.f_buffs,
            self.g_params, self.g_buffs,
            no_activation=self.no_activation
        )

        z = RevBlockFunction._backward(
            y.data,
            self.in_channels,
            self.out_channels,
            self.f_params, self.f_buffs,
            self.g_params, self.g_buffs,
            self.training,
            self.no_activation
        )

        self.assertEqual(self.x, z)


if __name__ == '__main__':
    unittest.main()
