# Some of this code is inspired by pytorch examples

import torch.nn.functional as F
import torch
from .basealgo import BaseAlgo


class Vae(BaseAlgo):
    def __init__(self, model, optimizer, mode='MNIST', RP=True):
        super().__init__(model, mode)
        self.optimizer = optimizer
        self.RP = RP

    def forward(self, X):
        (sample, mu, sigma), (model_sample, model_mu, model_sigma) = self.model(X, reparametrize=self.RP)
        return sample, mu, 2 * torch.log(sigma), model_sample, model_mu, 2 * torch.log(model_sigma)

    def get_loss(self, model_mu, mu, logvar, input):
        if self.mode == 'MNIST':
            BCE = F.binary_cross_entropy(model_mu, input, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return (BCE + KLD)/input.shape[0]
        else:
            # TODO
            pass

    def train_step(self, data):
        sample, mean, logvar, p_sample, p_mu, p_logvar = self.forward(data)
        loss = self.get_loss(p_mu, mean, logvar, data)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return mean, logvar, loss

    def test_step(self, data):
        with torch.no_grad():
            sample, mean, logvar, p_sample, p_mu, p_logvar = self.forward(data)
            loss = self.get_loss(p_mu, mean, logvar, data)

            return mean, logvar, p_mu, loss

