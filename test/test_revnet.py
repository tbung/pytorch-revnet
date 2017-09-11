import visualize
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from models.revnet import RevNet

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

test = RevNet([3, 3, 3], [16, 16, 32, 64], [1, 1, 1], 10)
dataiter = iter(trainloader)
images, labels = dataiter.next()
output = test(Variable(images))
g = visualize.make_dot(output)
g.view()
