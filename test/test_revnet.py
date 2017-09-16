from collections import OrderedDict
import visualize
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable, gradcheck
from models.revnet import RevNet, residual, RevBlockFunction

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

test = RevNet([3], [16, 16], [2], 10)
dataiter = iter(trainloader)
images, labels = dataiter.next()
output = test(Variable(images))
_, predicted = torch.max(output.data, 1)
print(predicted)
# g = visualize.make_dot(output)
# g.view()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(test.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0.0001)

loss = criterion(output, Variable(labels))
loss.backward()
# optimizer.step()

# crit = nn.L1Loss()
# test = Variable(torch.rand(3,3,3,3))
# target = Variable(torch.rand(3,3,3,3))

# f_params = OrderedDict()
# g_params = OrderedDict()

# in_channels = 3
# out_channels = 3
# f_params['w1'] = nn.Parameter(torch.Tensor(out_channels,
#                                in_channels, 3, 3))
# f_params['b1'] = nn.Parameter(torch.Tensor(out_channels))
# f_params['bw1'] = nn.Parameter(torch.Tensor(out_channels))
# f_params['bb1'] = nn.Parameter(torch.Tensor(out_channels))
# f_params['w2'] = nn.Parameter(torch.Tensor(out_channels,
#                                out_channels, 3, 3))
# f_params['b2'] = nn.Parameter(torch.Tensor(out_channels))
# f_params['bw2'] = nn.Parameter(torch.Tensor(out_channels))
# f_params['bb2'] = nn.Parameter(torch.Tensor(out_channels))

# g_params['w1'] = nn.Parameter(torch.Tensor(out_channels,
#                                out_channels, 3, 3))
# g_params['b1'] = nn.Parameter(torch.Tensor(out_channels))
# g_params['bw1'] = nn.Parameter(torch.Tensor(out_channels))
# g_params['bb1'] = nn.Parameter(torch.Tensor(out_channels))
# g_params['w2'] = nn.Parameter(torch.Tensor(out_channels,
#                                out_channels, 3, 3))
# g_params['b2'] = nn.Parameter(torch.Tensor(out_channels))
# g_params['bw2'] = nn.Parameter(torch.Tensor(out_channels))
# g_params['bb2'] = nn.Parameter(torch.Tensor(out_channels))

# f_rm1 = torch.zeros(out_channels)
# f_rv1 = torch.ones(out_channels)

# f_rm2 = torch.zeros(out_channels)
# f_rv2 = torch.ones(out_channels)

# g_rm1 = torch.zeros(out_channels)
# g_rv1 = torch.ones(out_channels)

# g_rm2 = torch.zeros(out_channels)
# g_rv2 = torch.ones(out_channels)

# optimizer = optim.SGD(list(f_params.values()), lr=0.1)

# res = RevBlockFunction.forward(test, *f_params.values(), f_rm1, f_rv1, f_rm2, f_rv2,
#         training=False)

# loss = crit(res, target)

# print(f_params['w2'].grad)

# loss.backward()

# print(f_params['w2'].grad)

# optimizer.step()
