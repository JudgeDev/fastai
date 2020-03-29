""" Refactored forward and backward

Convert forward_ and_backward into class:
PyCharm has no tool for this so it must be done manually
add class layer:
[store any parameters in __init__(self, params)]
turn def layer(x) into __call__(self, x)
store self.x
make self.y equal to old return value
return self.y
move layer_grad(inp, out) to class
change to backward(self)

instantiate layer in forward and backward - layer = Layer()
call layer.backward()
"""

# standard imports
# from pathlib import Path
# from IPython.core.debugger import set_trace
import pickle
import gzip
import math

# third party imports
from fastai import datasets
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import torch
from torch import tensor

MNIST_URL = 'http://deeplearning.net/data/mnist/mnist.pkl'


def get_data():
    """ Get data """
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
        return map(tensor, (x_train, y_train, x_valid, y_valid))


def normalize(data, mean, std):
    """ Normalise data """
    return (data - mean) / std


class Lin:
    """ Linear class """
    def __init__(self, w, b):
        self.w, self.b = w, b

    def __call__(self, x):
        self.x = x
        self.y = x @ self.w + self.b
        return self.y

    def backward(self):
        """ Grad of matmul with respect to input """
        self.x.g = self.y.g @ self.w.t()
        self.w.g = (self.x.unsqueeze(-1) * self.y.g.unsqueeze(1)).sum(0)
        self.b.g = self.y.g.sum(0)


class Relu:
    """ Relu class """
    def __call__(self, x):
        self.x = x
        self.y = x.clamp_min(0.)
        return self.y

    def backward(self):
        """" Grad of relu with respect to input activations """
        self.x.g = (self.x > 0).float() * self.y.g


class Mse:
    """ Mean squared error loss function class """
    def __call__(self, x, targ):
        self.x = x
        self.targ = targ
        self.y = (x.squeeze(-1) - targ).pow(2).mean()
        return self.y

    def backward(self):
        """ Grad of loss with respect to output of previous layer """
        self.x.g = 2. * (self.x.squeeze() - self.targ).unsqueeze(-1) / self.targ.shape[0]


def forward_and_backward(inp, targ):
    """ Forward and backward pass """
    relu = Relu()
    lin1 = Lin(w1, b1)
    lin2 = Lin(w2, b2)
    mse = Mse()
    # forward pass:
    l1 = lin1(inp)
    l2 = relu(l1)
    out = lin2(l2)
    # we don't actually need the loss in backward!
    loss = mse(out, targ)
    print(loss)

    # backward pass:
    mse.backward()
    lin1.backward()
    relu.backward()
    lin2.backward()


x_t, y_t, x_v, y_v = get_data()

# normalise data - use training, not validation mean for validation set
t_mean, t_std = x_t.mean(), x_t.std()
x_t = normalize(x_t, t_mean, t_std)
x_v = normalize(x_v, t_mean, t_std)

# size parameters
n, m = x_t.shape  # number and length of training inputs
c = y_t.max()+1  # number of classes
nh = 50  # number hidden layers

# kaiming init / he init for relu
w1 = torch.randn(m, nh)/math.sqrt(2 / m)
b1 = torch.zeros(nh)
w2 = torch.randn(nh, 1)/math.sqrt(nh)
b2 = torch.zeros(1)

forward_and_backward(x_t, y_t)
