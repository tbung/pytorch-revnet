import torch
import torch.autograd
from torch.autograd import Variable

# import visualize

import models.revnet as revnet

import unittest

from .common import TestCase


class TestRevNet(TestCase):
    def setUp(self):
        self.input = Variable(torch.rand(3, 4, 3, 3), requires_grad=True)
        self.in_channels = 4
        self.out_channels = 4
        self.training = True
        self.no_activation = True
        self.net = revnet.RevBlock(self.in_channels, self.out_channels)
        self.output = revnet.RevBlockFunction._inner(
                                        self.input,
                                        self.in_channels,
                                        self.out_channels,
                                        self.training,
                                        1,
                                        self.net.f_params,
                                        self.net._buffers,
                                        self.net.g_params,
                                        self.net._buffers,
                                        manual_grads=False,
                                        no_activation=self.no_activation)
        self.rec_input = revnet.RevBlockFunction._inner_backward(
                                        self.output.data,
                                        self.net.f_params,
                                        self.net._buffers,
                                        self.net.g_params,
                                        self.net._buffers,
                                        self.training, self.no_activation)
        # g = visualize.make_dot(self.output)
        # g.view()

    def test_grad(self):
        auto_grad = torch.autograd.grad(self.output, [self.input] +
                                        self.net.f_params 
                                        + self.net.g_params,
                                        Variable(torch.ones(3, 4, 3, 3),
                                                 requires_grad=True))

        manual_grad = revnet.RevBlockFunction._inner_grad(
                                    self.input.data,
                                    Variable(torch.ones(3, 4, 3, 3)),
                                    self.in_channels,
                                    self.out_channels,
                                    self.training,
                                    1,
                                    self.net.f_params,
                                    self.net._buffers,
                                    self.net.g_params,
                                    self.net._buffers,
                                    no_activation=self.no_activation)

        manual_grad = [manual_grad[0]] + [v for sub in manual_grad[1:]
                                          for v in sub]
        for auto, manual in zip(auto_grad, manual_grad):
            self.assertEqual(auto, manual)

    def test_recreation(self):
        self.assertEqual(self.input.data, self.rec_input)


if __name__ == '__main__':
    unittest.main()
