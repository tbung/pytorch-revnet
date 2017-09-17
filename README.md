# revnet

[PyTorch](http://pytorch.org/) implementation of [the reversible residual
network](https://arxiv.org/abs/1707.04585).

## Requirements

The main requirement ist obviously [PyTorch](http://pytorch.org/). CUDA is
strongly recommended.

The training script requires [tqdm](https://pypi.python.org/pypi/tqdm) for the
progress bar.

The unittests require the TestCase implemented by the PyTorch project. The
module can be downloaded
[here](https://github.com/pytorch/pytorch/blob/master/test/common.py).
