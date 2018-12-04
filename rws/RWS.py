import torch
from torch import nn

from rws.model import BasicModel


class RWS_1 (nn.Module):
    '''
    INPUT ARGUMENTS

    input_dim : input dimension (eg 784 for mini MNSIT)
    K : number of particles
    hidden_dim : hidden dimension (default = 200)
    hidden_layers : number of hidden layers (as described in BasicModel)
    encoding_dim : latent code dimension (default = 50)
    hidden_nonlinearity : hidden non linearity (default = tanh)
    decoder_nonlinearity : decoder non linearity (default = sigmoid)

    OUTPUT :

    RWS algo object with a model attribute
    train step : performs model update, q wake update q sleep update

    '''
    def __init__(self,input_dim , K=1,hidden_dim=200, hidden_layers=1, encoding_dim=50, hidden_nonlinearity='tanh',
                 decoder_nonlinearity='sigmoid'):
        super(RWS_1, self).__init__()
        self.model = BasicModel(input_dim,hidden_dim,hidden_layers,encoding_dim,hidden_nonlinearity,decoder_nonlinearity)
        self.K = K
    def forward(self,X):
        (sample, mu, sigma), p = self.model(X)
        return sample, mu, 2*torch.log(sigma),p

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
        h = torch.rand_like(mean)
        x_gh = self.model.decode(h)
        sample, mean, sigma = self.model.encode(x_gh)
        logvar = 2*torch.log(sigma)

        log_q_h_gx = torch.sum(-0.5*logvar -0.5*torch.exp(-logvar)*(h-mean)**2,-1)
        return -torch.mean(log_q_h_gx)


    def get_importance_weight_gauss(self, mean,logvar,input):
        eps = torch.rand_like(mean)
        h = mean + eps*torch.exp(logvar/2)
        h = h.detach()
        x_gh = self.model.decode(h)
        log_q_h_gx = torch.sum(-0.5*logvar -0.5*torch.exp(-logvar)*(h-mean)**2,-1)
        log_p_x_gh = torch.sum(input*torch.log(x_gh) + (1-input)*torch.log(1-x_gh), -1)
        log_p_h = torch.sum(-0.5*(h)**2,-1)

        log_p_x_h = log_p_x_gh + log_p_h
        log_weight = log_p_x_gh + log_p_h - log_q_h_gx
       # weight = torch.exp(log_weight)


        return log_weight, log_q_h_gx, log_p_x_h

    def train_step(self,optim_model, optim_recog, data):

        #model update
        sample, mean, logvar, p = self.forward(data)
        loss_model = self.get_loss_p_update(mean, logvar, data)
        self.zero_grad()
        loss_model.backward()
        optim_model.step()

        sample, mean, logvar, p   = self.forward(data)

        #Wake Q update
        self.zero_grad()
        loss_q_wake = self.get_loss_q_wake_update(mean, logvar, data)
        loss_q_wake.backward()
        optim_recog.step()

        #Sleep q update
        self.zero_grad()
        loss_q_sleep = self.get_loss_q_sleep_update(mean)
        loss_q_sleep.backward()
        optim_recog.step()

        return mean, logvar, loss_model,loss_q_wake, loss_q_sleep












