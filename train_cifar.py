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
import models.revnet as revnet


parser = argparse.ArgumentParser()
parser.add_argument("--model", metavar="NAME",
                    help="what model to use")
parser.add_argument("--load", metavar="PATH",
                    help="load a previous model state")
parser.add_argument("-e", "--evaluate", action="store_true",
                    help="evaluate model on validation set")
parser.add_argument("--batch-size", default=128, type=int,
                    help="size of the mini-batches")
parser.add_argument("--epochs", default=200, type=int,
                    help="number of epochs")
parser.add_argument("--lr", default=0.1, type=float,
                    help="initial learning rate")


# Check if CUDA is avaliable
CUDA = torch.cuda.is_available()

best_acc = 0


def main():
    global best_acc

    args = parser.parse_args()

    model = getattr(models, args.model)()

    if CUDA:
        model.cuda()

    if args.load is not None:
        load(model, args.load)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=0.0001)
    # step_size = args.epochs // 
    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150],
                            gamma=0.1)

    # Load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform_train
    )

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform_test
    )

    valloader = torch.utils.data.DataLoader(testset,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    if args.evaluate:
        validate(model)
        return

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        train(epoch, model, criterion, optimizer, trainloader)
        acc = validate(model, valloader)

        if acc > best_acc:
            best_acc = acc
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            save_checkpoint(model)
        print('Accuracy: {}%'.format(acc))


def train(epoch, model, criterion, optimizer, trainloader):
    # model.train()
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
        revnet.free()
        optimizer.step()


def validate(model, valloader):
    correct = 0
    total = 0

    # model.eval()

    for data in valloader:
        images, labels = data
        if CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    acc = 100 * correct / total

    return acc


def load(model, path):
    if path == "latest":
        load_latest(model)
    else:
        model.load_state_dict(torch.load(path))


def load_latest(model):
    checkpoints = glob.glob('./checkpoints/*{}*'.format(model.name))
    latest = max(checkpoints, key=os.path.getctime)
    load(model, latest)


def save_checkpoint(model):
    path = "./checkpoints/cifar_{0}_{1:%Y-%m-%d}_{1:%H-%M-%S}.dat"
    torch.save(model.state_dict(),
               path.format(model.name, datetime.now()))


if __name__ == "__main__":
    main()
