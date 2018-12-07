import numpy as np
from data import gmm_gen
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical

class exp:
    def __init__(self, gmm_toy, model):
        self.toy = gmm_toy
        self.model = model
        self.mode = model.mode
        self.pi = torch.from_numpy(gmm_toy.latent_proba).float()
        self.mus = gmm_toy.mus
        self.cov = gmm_toy.cov

    def gradients_recog(self, data, algo):
        return

    def gradients_model(self, data, algo):
        return

    def inference_perf(self, data):
        if self.mode == 'dis-GMM':
            C = self.toy.C
            log_p_x_hs = torch.zeros((data.size()[0], C))
            sample, probas, _ = self.model.encode(data)
            h = OneHotCategorical(probas).sample()
            log_q_h_gx = torch.sum(h * torch.log(probas))

            x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)
            log_p_x_gh = torch.sum(
                -0.5 * torch.log(x_gh_sigma ** 2) - 0.5 * (x_gh_sigma ** 2) * (data - x_gh_mean) ** 2, -1)
            log_p_h = torch.sum(h * torch.log(self.pi), -1)

            log_p_x_h = log_p_x_gh + log_p_h


            q_z_gx = torch.exp(log_q_h_gx)

            for i in range(C):

                probas = torch.eye(C)[i,:]
                h = OneHotCategorical(probas).sample((data.size()[0],))
                x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)
                log_p_x_gh = torch.sum(
                    -0.5 * torch.log(x_gh_sigma ** 2) - 0.5 * (x_gh_sigma ** 2) * (data - x_gh_mean) ** 2, -1)
                log_p_h = torch.sum(h * torch.log(self.pi), -1)

                log_p_x_h = log_p_x_gh + log_p_h

                log_p_x_hs[:,i] = log_p_x_h

            log_w_max, _ = torch.max(log_p_x_hs, 1)

            diff = log_p_x_hs - log_w_max.unsqueeze(1)
            log_sum = torch.log(torch.sum(torch.exp(diff), -1))
            denom_log = log_w_max + log_sum
            log_p_z_gx = log_p_x_h - denom_log
            diff = torch.exp(log_p_z_gx) - q_z_gx

        return torch.mean((diff)**2)

    def model_perf(self, data):
        if self.mode =='dis-GMM':
            diff_pi = torch.mean(torch.abs(self.pi - self.model.pi))
            diff_mu = torch.mean((self.mus -self.model.mus)**2)

            return diff_pi

        else:
            raise NotImplementedError('mode not implemented')

