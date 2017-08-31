import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import models

# Check if CUDA is avaliable
CUDA = torch.cuda.is_available()

model = models.resnet32()

if CUDA:
    model.cuda()


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


def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1,
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
            outputs = model(inputs)
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

    # # plot training loss
    # plt.figure()
    # plt.plot(range(len(losses)), losses)
    # plt.savefig('loss.pdf')

    torch.save(model.state_dict(), "cifar_{}.dat".format("test"))

    print('Finished Training')


def load(path):
    model.load_state_dict(torch.load(path))


def test():
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        if CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = model(Variable(images))
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
        outputs = model(Variable(images))
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
