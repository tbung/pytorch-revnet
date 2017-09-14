from .resnet import ResNet
from .revnet import RevNet

__all__ = ["ResNet", "resnet32", "resnet34", "revnet38"]


def resnet32():
    model = ResNet(
            units=[5, 5, 5],
            filters=[16, 16, 32, 64],
            strides=[1, 2, 2],
            classes=10
            )
    model.name = "resnet32"
    return model

def resnet34():
    model = ResNet(
            units=[3, 4, 6, 3],
            filters=[64, 64, 128, 256, 512],
            strides=[1, 2, 2, 2],
            classes=10
            )
    model.name = "resnet34"
    return model


def resnet56():
    model = ResNet(
            units=[9, 9, 9],
            filters=[16, 16, 32, 64],
            strides=[1, 2, 2],
            classes=10
            )
    model.name = "resnet56"
    return model


def revnet38():
    model = RevNet(
            units=[3, 3, 3],
            filters=[32, 32, 64, 112],
            strides=[1, 2, 2],
            classes=10
            )
    model.name = "revnet38"
    return model
