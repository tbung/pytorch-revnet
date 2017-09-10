import test.test_revnet as t
import visualize
from torch.autograd import Variable
from models.revnet import RevNet

test = RevNet([3, 3, 3], [16, 16, 32, 64], [1, 1, 1], 10)
dataiter = iter(t.trainloader)
images, labels = dataiter.next()
output = test(Variable(images))
g = visualize.make_dot(output)
g.view()
