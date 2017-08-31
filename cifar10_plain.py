import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt


# Check if CUDA is avaliable
CUDA = torch.cuda.is_available()


# Set network depth
# Network will have 6*n + 2 weighted layers
n = 5


# Prepare our data set
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class PlainNet(nn.Module):
    """
    Implementation of the plain convnet as described in
    https://arxiv.org/abs/1512.03385
    """
    def __init__(self, n):
        super(PlainNet, self).__init__()

        self.n = n

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(3, 16, 3, padding=1))

        self.layers.append(nn.BatchNorm2d(16))

        for i in range(2 * self.n):
            self.layers.append(nn.Conv2d(16, 16, 3, padding=1))
            self.layers.append(nn.BatchNorm2d(16))

        self.layers.append(nn.Conv2d(16, 32, 3, stride=2, padding=1))
        self.layers.append(nn.BatchNorm2d(32))

        for i in range(2 * self.n - 1):
            self.layers.append(nn.Conv2d(32, 32, 3, padding=1))
            self.layers.append(nn.BatchNorm2d(32))

        self.layers.append(nn.Conv2d(32, 64, 3, stride=2, padding=1))
        self.layers.append(nn.BatchNorm2d(64))

        for i in range(2 * self.n - 1):
            self.layers.append(nn.Conv2d(64, 64, 3, padding=1))
            self.layers.append(nn.BatchNorm2d(64))

        self.layers.append(nn.AvgPool2d(8))

        self.layers.append(nn.Linear(64, 10))

    def forward(self, x):
        for i in range(0, 12 * self.n + 2, 2):
            x = self.layers[i](x)
            x = F.relu(self.layers[i+1](x))

        x = self.layers[12 * self.n + 2](x)

        x = x.view(-1, 64)

        x = F.softmax(self.layers[12 * self.n + 3](x))

        return x


net = PlainNet(n)

if CUDA:
    net.cuda()


def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1,
                          momentum=0.9, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    losses = []

    for epoch in range(80):  # loop over the dataset multiple times
        scheduler.step()

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if CUDA:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.data[0])

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[{0}, {1}] loss: {2:.4f}'.format(
                      epoch + 1, i + 1, running_loss / 100
                      ))
                running_loss = 0.0

    # plot training loss
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.savefig('loss.pdf')

    torch.save(net.state_dict(), "cifar10_plain{}.dat".format(6*n + 2))

    print('Finished Training')


def load(path):
    net.load_state_dict(torch.load(path))


def test():
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        if CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: {}\%'.format(
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        if CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of {0} : {1}\%'.format(
            classes[i], 100 *
            class_correct[i] /
            class_total[i]))


if __name__ == "__main__":
    train()
    test()
