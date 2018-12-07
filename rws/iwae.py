import torch
from torch.distributions import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical
import numpy as np

from torch import nn

from rws.model import BasicModel


class IWAE:
    '''
    INPUT ARGUMENTS

    model : the model to use
    K : number of particles
    optim : optimizer
    mode : 'MNIST', 'dis-GMM', 'cont-GMM'
    RP : use reparametrization trick or not

    OUTPUT :

    RWS algo object with a model attribute
    train step : performs model update, q wake update q sleep update

    '''

    def __init__(self, model, optim, K=1, mode='MNIST', RP=False, VR=None):
        self.model = model
        self.optim = optim
        self.K = K
        self.mode = mode
        self.RP = RP
        self.VR = VR

    def forward(self, X):
        (sample, mu, sigma), (model_sample, model_mu, model_sigma) = self.model(X, reparametrize =self.RP)
        return sample, mu, 2 * torch.log(sigma), model_sample, model_mu, 2 * torch.log(model_sigma)

    def get_loss(self, mean, logvar, input):

        log_weights = torch.zeros((input.size()[0], self.K))
        log_ps = torch.zeros((input.size()[0], self.K))
        log_qs = torch.zeros((input.size()[0], self.K))

        for i in range(self.K):
            log_weight, log_q, log_p = self.get_importance_weight(mean, logvar, input)

            log_weights[:, i] = log_weight
            log_ps[:, i] = log_p
            log_qs[:, i] = log_q

        log_w_max, _ = torch.max(log_weights, 1)

        diff = log_weights - log_w_max.unsqueeze(1)

        log_sum = torch.log(torch.sum(torch.exp(diff), 1))
        denom_log = log_w_max + log_sum

        logs = log_weights - denom_log.unsqueeze(1)
        weights = torch.exp(logs).detach()
        batch_loss = torch.sum((log_ps - log_qs) * weights, -1)
        loss = - torch.mean(batch_loss)
        reinf_loss = self.get_reinforce_loss(denom_log, log_qs, log_weights, log_w_max)
        return loss + reinf_loss

    def get_reinforce_loss(self, denom_log, log_qs, log_weights, log_w_max):
        if self.RP:
            return 0
        if self.VR is None:
            learning_signal = (denom_log - torch.log(torch.tensor(self.K).float())).detach()  # n
            return - torch.mean(learning_signal * torch.sum(log_qs, 1))
        if self.VR == 'VIMCO':
            fake_log_weights = torch.sum(log_weights, 1).unsqueeze(1)
            fake_log_weights = - (log_weights - fake_log_weights) / (self.K - 1)  # n x k
            fake_diff = fake_log_weights - log_w_max.unsqueeze(1)  # n x k
            log_w_bar = torch.transpose(log_weights.repeat((self.K, 1, 1)), 0, 1)  # n x k x k
            log_w_bar[:, range(self.K), range(self.K)] = fake_log_weights
            diff_bar = log_w_bar - log_w_max.unsqueeze(1).unsqueeze(1)  # n x k x k

            log_sum_bar = torch.log(torch.sum(torch.exp(diff_bar), 2))  # n x k
            denom_log_bar = log_sum_bar + log_w_max.unsqueeze(1)  # n x k
            learning_signal = - (denom_log_bar - denom_log.unsqueeze(1))  # n x k
            return - torch.mean(torch.sum(log_qs * learning_signal, 1))

    def get_nll(self,  input, K):
        sample, mean, logvar, _, _,_  = self.forward(input)
        log_weights = torch.zeros((input.size()[0], K))
        log_ps = torch.zeros((input.size()[0],K))
        log_qs = torch.zeros((input.size()[0], K))

        for i in range(K):
            log_weight, log_q, log_p = self.get_importance_weight(mean, logvar, input)

            log_weights[:, i] = log_weight
            log_ps[:, i] = log_p
            log_qs[:, i] = log_q

        log_w_max, _ = torch.max(log_weights, 1)

        diff = log_weights - log_w_max.unsqueeze(1)

        log_sum = torch.log(torch.sum(torch.exp(diff), 1))
        denom_log = log_w_max + log_sum

        logs = log_weights - denom_log.unsqueeze(1)
        weights = torch.exp(logs).detach()
        batch_loss = torch.sum((log_ps - log_qs) * weights, -1)
        return -torch.mean(batch_loss)

    def get_importance_weight(self, mean, logvar, input):

        if self.mode == 'MNIST':
            if not self.RP:
                h = Normal(mean, torch.exp(logvar / 2)).sample()
            if self.RP:
                eps = Normal(torch.zeros(mean.size()), torch.ones(logvar.size())).sample()
                h = mean + eps * torch.exp(logvar / 2)
            x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)
            h, x_gh_sample, x_gh_mean, x_gh_sigma, mean, logvar, input = h.squeeze(), x_gh_sample.squeeze(), x_gh_mean.squeeze(), x_gh_sigma.squeeze(), mean.squeeze(), logvar.squeeze(), input.squeeze()

            log_q_h_gx = torch.sum(-0.5 * logvar - 0.5 * torch.exp(-logvar) * (h - mean) ** 2, -1)

            log_p_x_gh = torch.sum(input * torch.log(x_gh_mean) + (1 - input) * torch.log(1 - x_gh_mean), -1)
            log_p_h = torch.sum(-0.5 * (h) ** 2, -1)

        if self.mode == 'dis-GMM':
            h = OneHotCategorical(mean).sample()
            x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)

            log_q_h_gx = torch.sum(h * torch.log(mean))
            log_p_x_gh = torch.sum(
                -0.5 * torch.log(x_gh_sigma ** 2) - 0.5 * (x_gh_sigma ** 2) * (input - x_gh_mean) ** 2, -1)
            log_p_h = torch.sum(h * torch.log(self.model.pi), -1)
        if self.mode == 'cont-GMM':
            if not self.RP:
                h = Normal(mean, torch.exp(logvar / 2)).sample()
            if self.RP:
                eps = Normal(torch.zeros(mean.size()), torch.ones(logvar.size())).sample()
                h = mean + eps * torch.exp(logvar / 2)
            x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)

            log_q_h_gx = torch.sum(-0.5 * logvar - 0.5 * torch.exp(-logvar) * (h - mean) ** 2, -1)
            log_p_x_gh = torch.sum(
                -0.5 * torch.log(x_gh_sigma ** 2) - 0.5 * (x_gh_sigma ** 2) * (input - x_gh_mean) ** 2, -1)
            log_p_h = torch.sum(-0.5 * (h) ** 2, -1)

        log_p_x_h = log_p_x_gh + log_p_h
        log_weight = log_p_x_gh + log_p_h - log_q_h_gx

        return log_weight, log_q_h_gx, log_p_x_h

    def train_step(self, data):

        # model update
        sample, mean, logvar, p_sample, p_mu, p_logvar = self.forward(data)
        loss = self.get_loss(mean, logvar, data)
        self.model.zero_grad()
        loss.backward()
        self.optim.step()

        if self.mode == 'MNIST':
            return mean, logvar, loss
        elif self.mode == 'dis-GMM':
            return loss

    def test_step(self, data):
        with torch.no_grad():
            sample, mean, logvar, p_sample, p_mu, p_logvar = self.forward(data)
            loss = self.get_loss(mean, logvar, data)

            if self.mode == 'MNIST':
                return mean, logvar, p_mu, loss
            elif self.mode == 'dis-GMM':
                return loss

    def visu(self, writer, step, args):
        mean, logvar, loss = args
        if self.mode == 'MNIST':
            eps = torch.rand_like(mean)
            h = mean + eps * torch.exp(logvar / 2)
            out, _, _ = self.model.decode(eps)
            out = out.view(mean.size()[0], 1, 28, 28)
            rec, _, _ = self.model.decode(h)
            rec = rec.view(mean.size()[0], 1, 28, 28)
            writer.add_scalar('loss', loss, step)
            writer.add_image('im_0', out[0], step)
            writer.add_image('im_1', out[1], step)
            writer.add_image('rec_0', rec[2], step)
            writer.add_image('rec_1', rec[3], step)
