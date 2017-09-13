import visualize
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable, gradcheck
from models.revnet import RevNet, RevBlock, RevGroupFunction

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
# g = visualize.make_dot(output)
# g.view()

# gradchek takes a tuple of tensor as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (Variable(torch.rand(4,16,32,32), requires_grad=True),
         nn.ModuleList([RevBlock(16, 16), RevBlock(16, 16)]))
# test = gradcheck(RevGroupFunction.apply, input, eps=1e-6, atol=1e-4)
test = RevGroupFunction.apply(*input)
print(test)
