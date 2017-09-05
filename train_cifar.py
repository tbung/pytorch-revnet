from datetime import datetime

import glob
import os
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import models

# Check if CUDA is avaliable
CUDA = torch.cuda.is_available()

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


# losses = []


def train(epoch, model, criterion, optimizer):
        for i, data in enumerate(tqdm(trainloader,
                                      ascii=True,
                                      desc='{:03d}'.format(epoch))):
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

            # print statistics
            # running_loss += loss.data[0]
            # if i % 100 == 99:    # print every 2000 mini-batches
            #     print('[{0}, {1}] loss: {2:.4f}'.format(
            #           epoch + 1, i + 1, running_loss / 100
            #           ))
            #     running_loss = 0.0

    # # plot training loss
    # plt.figure()
    # plt.plot(range(len(losses)), losses)
    # plt.savefig('loss.pdf')


def load(model, path):
    if path == "latest":
        load_latest(model)
    else:
        model.load_state_dict(torch.load(path))


def load_latest(model):
    checkpoints = glob.glob('./checkpoints/*{}*'.format(model.name))
    latest = max(checkpoints, key=os.path.getctime)
    load(model, latest)


def checkpoint(model):
    if CUDA:
        model.cpu()
    torch.save(model.state_dict(),
               "./checkpoints/cifar_{0}_{1:%Y-%m-%d}_{1:%H-%M-%S}.dat".format(model.name,
               datetime.now()))
    if CUDA:
        model.cuda()


best_acc = 0


def test(model):
    global best_acc
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

    acc = 100 * correct / total

    if acc > best_acc:
        best_acc = acc
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        checkpoint(model)
    print('Accuracy: {}%'.format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--load")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    model = getattr(models, args.model)()

    if CUDA:
        model.cuda()

    if args.load is not None:
        load(model, args.load)

    if not args.test:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1,
                              momentum=0.9, weight_decay=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[150, 250, 350],
                                gamma=0.1)

        for epoch in range(350):  # loop over the dataset multiple times
            scheduler.step()
            train(epoch, model, criterion, optimizer)
            test(model)

        load_latest(model)
        test(model)
    else:
        test(model)
