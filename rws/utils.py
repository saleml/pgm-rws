# Some of this code is copied from https://github.com/dev4488/VAE_gumble_softmax/blob/master/vae_gumbel_softmax.py

import torch
from torch.nn import functional as F
import numpy as np


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    print(y_hard.scatter_(1, ind.view(-1, 1), 1))
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(logits.shape[0], -1)


print(gumbel_softmax(torch.log(torch.tensor([0.4, 0.6, 0.8, 0.2, 0.1, 0.9, 0.3, 0.7]).view(2, 4)), .1, ))