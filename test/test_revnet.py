import visualize
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from resnet import ResNet18
import torchvision.transforms as transforms
from torch.autograd import Variable, gradcheck
from models.revnet import RevNet, RevBlock, RevGroupFunction
from models import revnet38
# from models.resnet import ResNet

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# test = RevNet([3], [16, 16], [2], 10)
test = ResNet18()
dataiter = iter(trainloader)
images, labels = dataiter.next()
output = test(Variable(images))
_, predicted = torch.max(output.data, 1)
print(predicted)
# g = visualize.make_dot(output)
# g.view()
images, labels = dataiter.next()
output = test(Variable(images))
_, predicted = torch.max(output.data, 1)
print(predicted)
images, labels = dataiter.next()
output = test(Variable(images))
_, predicted = torch.max(output.data, 1)
print(predicted)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(test.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0.0001)

loss = criterion(output, Variable(labels))
# print(test.groups[0][1].f.conv1.bias)
# print(test.groups[0][1].f.conv1.bias.grad)
w = [torch.Tensor(i.data) for i in list(test.parameters())]
loss.backward()
# print(test.groups[0][1].f.conv1.bias)
# print(test.groups[0][1].f.conv1.bias.grad)
# print(test.layers[2].conv1.bias)
optimizer.step()
# print(test.groups[0][1].f.conv1.bias)
# print(test.groups[0][1].f.conv1.bias.grad)
output = test(Variable(images))
_, predicted = torch.max(output.data, 1)
print(predicted)
images, labels = dataiter.next()
output = test(Variable(images))
_, predicted = torch.max(output.data, 1)
print(predicted)
images, labels = dataiter.next()
output = test(Variable(images))
_, predicted = torch.max(output.data, 1)
print(predicted)
n, params = zip(*list(test.named_parameters()))
for i, p in enumerate(params):
    if not w[i].equal(p.data):
        print(n[i])
