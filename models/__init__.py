from .resnet import ResNet

__all__ = ["ResNet", "resnet32"]


def resnet32():
    model = ResNet(
            units=[5, 5, 5],
            filters=[16, 16, 32, 64],
            strides=[1, 2, 2],
            classes=32
            )
    return model
