from datetime import datetime

import os
import sys
import argparse

from tqdm import tqdm

# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

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
parser.add_argument("--clip", default=0, type=float,
                    help="maximal gradient norm")
parser.add_argument("--weight-decay", default=1e-4, type=float,
                    help="weight decay factor")
parser.add_argument("--stats", action="store_true",
                    help="record and plot some stats")


# Check if CUDA is avaliable
CUDA = torch.cuda.is_available()

best_acc = 0


def main():
    global best_acc

    args = parser.parse_args()

    model = getattr(models, args.model)()

    exp_id = "cifar_{0}_{1:%Y-%m-%d}_{1:%H-%M-%S}".format(model.name,
                                                          datetime.now())

    path = os.path.join("./experiments/", exp_id, "cmd.sh")
    if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        f.write(' '.join(sys.argv))

    if CUDA:
        model.cuda()

    if args.load is not None:
        load(model, args.load)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr*10,
                          momentum=0.9, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    print("Prepairing data...")

    # Load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
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
        print("\nEvaluating model...")
        acc = validate(model, valloader)
        print('Accuracy: {}%'.format(acc))
        return

    if args.stats:
        losses = []
        taccs = []
        vaccs = []

    print("\nTraining model...")
    for epoch in range(args.epochs):
        scheduler.step()
        loss, train_acc = train(epoch, model, criterion, optimizer,
                                trainloader, args.clip)
        val_acc = validate(model, valloader)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, exp_id)
        print('Accuracy: {}%'.format(val_acc))

        if args.stats:
            losses.append(loss)
            taccs.append(train_acc)
            vaccs.append(val_acc)

    save_checkpoint(model, exp_id)

    if args.stats:
        path = os.path.join("./experiments/", exp_id, "stats/{}.dat")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path.format('loss'), 'w') as f:
            for i in losses:
                f.write('{}\n'.format(i))

        with open(path.format('taccs'), 'w') as f:
            for i in taccs:
                f.write('{}\n'.format(i))

        with open(path.format('vaccs'), 'w') as f:
            for i in vaccs:
                f.write('{}\n'.format(i))


def train(epoch, model, criterion, optimizer, trainloader, clip):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    t = tqdm(trainloader, ascii=True, desc='{}'.format(epoch).rjust(3))
    for i, data in enumerate(t):
        inputs, labels = data

        if CUDA:
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Free the memory used to store activations
        if type(model) is revnet.RevNet:
            revnet.free()

        if clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        acc = 100 * correct / total

        t.set_postfix(loss='{:.3f}'.format(train_loss/(i+1)).ljust(3),
                      acc='{:2.1f}%'.format(acc).ljust(6))

    return train_loss, acc


def validate(model, valloader):
    correct = 0
    total = 0

    model.eval()

    for data in valloader:
        images, labels = data
        if CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = model(Variable(images))

        # Free the memory used to store activations
        if type(model) is revnet.RevNet:
            revnet.free()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    acc = 100 * correct / total

    return acc


def load(model, path):
    model.load_state_dict(torch.load(path))


def save_checkpoint(model, exp_id):
    path = os.path.join(
        "experiments", exp_id, "checkpoints",
        "cifar_{0}_{1:%Y-%m-%d}_{1:%H-%M-%S}.dat".format(model.name,
                                                         datetime.now()))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
