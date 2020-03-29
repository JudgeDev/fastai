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


def lin(x, w, b):
    """ Linear layer """
    return x @ w + b


def relu(x):
    """ Relu layer """
    return x.clamp_min(0.)


def mse(output, targ):
    """ Mean squared error loss function """
    return (output.squeeze(-1) - targ).pow(2).mean()


def mse_grad(inp, targ):
    """ Grad of loss with respect to output of previous layer """
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]


def relu_grad(inp, out):
    """" Grad of relu with respect to input activations """
    inp.g = (inp > 0).float() * out.g


def lin_grad(inp, out, w, b):
    """ Grad of matmul with respect to input """
    inp.g = out.g @ w.t()
    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)
    b.g = out.g.sum(0)


def forward_and_backward(inp, targ):
    """ Forward and backward pass """
    # forward pass:
    l1 = lin(inp, w1, b1)
    l2 = relu(l1)
    out = lin(l2, w2, b2)
    # we don't actually need the loss in backward!
    loss = mse(out, targ)
    print(loss)

    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)


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
