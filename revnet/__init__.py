from .resnet import ResNet
from .revnet import RevNet, RevBlock, RevBlockFunction, possible_downsample


def resnet32():
    model = ResNet(
            units=[5, 5, 5],
            filters=[16, 16, 32, 64],
            strides=[1, 2, 2],
            classes=10
            )
    model.name = "resnet32"
    return model


def resnet110():
    model = ResNet(
            units=[18, 18, 18],
            filters=[16, 16, 32, 64],
            strides=[1, 2, 2],
            classes=10
            )
    model.name = "resnet110"
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


def revnet110():
    model = RevNet(
            units=[9, 9, 9],
            filters=[32, 32, 64, 112],
            strides=[1, 2, 2],
            classes=10
            )
    model.name = "revnet110"
    return model
