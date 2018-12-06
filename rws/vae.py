import torch.nn.functional as F
from torch.distributions import multivariate_normal
import torch


def reconstruction_loss(x, reconstruction, *args):
    return F.binary_cross_entropy(reconstruction, x, reduction='sum')


def normal_pdf_loss(x, mu, sigma, *args):
    normal = multivariate_normal(mu, sigma)
    return normal(x).sum()


class Vae(object):
    def __init__(self, model, optimizer, mode):
        self.model = model
        self.optimizer = optimizer
        if mode == 'MNIST':
            self.loss = reconstruction_loss
        else:
            self.loss = normal_pdf_loss

    def train_step(self, x):
        self.optimizer.zero_grad()
        (sample, mu, sigma), (_, x_mu, x_sigma) = self.model(x, reparametrize=True)
        generation = self.loss(x, x_mu, x_sigma)
        logvar = 2 * torch.log(sigma)
        variational = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = generation + variational
        loss.backward()
        self.optimizer.step()
        return loss

