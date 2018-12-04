import torch
from torch.distributions import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical
import numpy as np

from torch import nn

from rws.model import BasicModel


class RWS_1 (object):
    '''
    INPUT ARGUMENTS

    model : the model to use
    K : number of particles
    optim_recog : optimizer of recognition network
    optim_model : optimizer of model
    mode : 'MNIST', 'dis-GMM', 'cont-GMM'

    OUTPUT :

    RWS algo object with a model attribute
    train step : performs model update, q wake update q sleep update

    '''
    def __init__(self, model, optim_recog, optim_model, K=1, mode ='MNIST'):
        super(RWS_1, self).__init__()
        self.model = model
        self.optim_recog = optim_recog
        self.optim_model = optim_model
        self.K = K
        self.mode = mode
    def forward(self,X):
        (sample, mu, sigma), (model_sample, model_mu, model_sigma) = self.model(X)
        return sample, mu, 2*torch.log(sigma), model_sample, model_mu, 2*torch.log(model_sigma)

    def get_loss_p_update(self, mean, logvar, input):

        log_weights = torch.zeros((input.size()[0],self.K))
        log_ps = torch.zeros((input.size()[0],self.K))

        for i in range(self.K):
            log_weight, log_q, log_p = self.get_importance_weight_gauss(mean, logvar, input)
            log_weights[:,i] = log_weight
            log_ps[:,i] = log_p


        log_w_max,_ = torch.max(log_weights,1)

        diff = log_weights - log_w_max.unsqueeze(1)

        log_sum = torch.log(torch.sum(torch.exp(diff),1))
        denom_log = log_w_max+ log_sum

        logs = log_weights - denom_log.unsqueeze(1)
        weights = torch.exp(logs).detach()
        batch_loss = torch.sum(log_ps*weights,-1)
        return -torch.mean(batch_loss)


    def get_loss_q_wake_update(self,mean, logvar, input):

        log_weights = torch.zeros((input.size()[0], self.K))
        log_qs = torch.zeros((input.size()[0], self.K))

        for i in range(self.K):
            log_weight, log_q, log_p = self.get_importance_weight_gauss(mean, logvar, input)
            log_weights[:, i] = log_weight
            log_qs[:, i] = log_q

        log_w_max, _ = torch.max(log_weights, 1)

        diff = log_weights - log_w_max.unsqueeze(1)
        log_sum = torch.log(torch.sum(torch.exp(diff), -1))
        denom_log = log_w_max + log_sum

        logs = log_weights - denom_log.unsqueeze(1)
        weights = torch.exp(logs).detach()
        batch_loss = torch.sum(log_qs * weights, -1)
        return -torch.mean(batch_loss)

    def get_loss_q_sleep_update(self,mean):

        if self.mode == 'MNIST' :
            h = torch.rand_like(mean)
            x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)
            sample, mean, sigma = self.model.encode(x_gh_sample)
            logvar = 2*torch.log(sigma)

            log_q_h_gx = torch.sum(-0.5*logvar -0.5*torch.exp(-logvar)*(h-mean)**2,-1)

        if self.mode == 'dis-GMM':
            h = OneHotCategorical(self.model.pi)
            x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)
            sample, mean, sigma = self.model.encode(x_gh_sample)

            log_q_h_gx = torch.sum(h*torch.log(mean))

        if self.mode == 'cont-GMM':
            h = torch.rand_like(mean)
            x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)
            sample, mean, sigma = self.model.encode(x_gh_sample)

            log_q_h_gx = torch.sum(-0.5 * logvar - 0.5 * torch.exp(-logvar) * (h - mean) ** 2, -1)


        return -torch.mean(log_q_h_gx)


    def get_importance_weight(self, mean,logvar,input):

        if self.mode == 'MNIST':
            h = Normal(mean, torch.exp(logvar / 2))
            x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)

            log_q_h_gx = torch.sum(-0.5*logvar -0.5*torch.exp(-logvar)*(h-mean)**2,-1)
            log_p_x_gh = torch.sum(input*torch.log(x_gh_mean) + (1-input)*torch.log(1-x_gh_mean), -1)
            log_p_h = torch.sum(-0.5*(h)**2,-1)
        if self.mode == 'dis-GMM' :
            h = OneHotCategorical(mean)
            x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)

            log_q_h_gx = torch.sum(h*torch.log(mean))
            log_p_x_gh = torch.sum( -0.5*torch.log(x_gh_sigma**2) -0.5*(x_gh_sigma**2)*(input -x_gh_mean)**2, -1)
            log_p_h = torch.sum(h*torch.log(self.model.pi), -1)
        if self.mode == 'cont-GMM' :
            h = Normal(mean, torch.exp(logvar / 2))
            x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)

            log_q_h_gx = torch.sum(-0.5 * logvar - 0.5 * torch.exp(-logvar) * (h - mean) ** 2, -1)
            log_p_x_gh = torch.sum(
                -0.5 * torch.log(x_gh_sigma ** 2) - 0.5 * (x_gh_sigma ** 2) * (input - x_gh_mean) ** 2, -1)
            log_p_h = torch.sum(-0.5 * (h) ** 2, -1)




        log_p_x_h = log_p_x_gh + log_p_h
        log_weight = log_p_x_gh + log_p_h - log_q_h_gx

        return log_weight, log_q_h_gx, log_p_x_h

    def train_step(self, data):

        #model update
        sample, mean, logvar, p_sample, p_mu, p_logvar = self.forward(data)
        loss_model = self.get_loss_p_update(mean, logvar, data)
        self.model.zero_grad()
        loss_model.backward()
        self.optim_model.step()

        sample, mean, logvar, p_sample, p_mu, p_logvar  = self.forward(data)

        #Wake Q update
        self.model.zero_grad()
        loss_q_wake = self.get_loss_q_wake_update(mean, logvar, data)
        loss_q_wake.backward()
        self.optim_recog.step()

        #Sleep q update
        self.model.zero_grad()
        loss_q_sleep = self.get_loss_q_sleep_update(mean)
        loss_q_sleep.backward()
        self.optim_recog.step()

        return mean, logvar, loss_model,loss_q_wake, loss_q_sleep











