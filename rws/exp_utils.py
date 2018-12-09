import numpy as np
from data import gmm_gen
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
import seaborn as sns

class exp:
    def __init__(self, gmm_toy, model):
        self.toy = gmm_toy
        self.model = model
        self.mode = model.mode

        if gmm_toy is not None :
            self.mus = gmm_toy.mus
            self.cov = gmm_toy.cov
            self.pi = torch.from_numpy(gmm_toy.latent_proba).float()

    def gradients_recog(self, data, algo, algo_ = 'rws', K =5):
        algo.K = K
        if algo_ == 'rws':
            grads = []
            for i in range(50):
                sample, mean, logvar, _, _, _=algo.forward(data[0:1,:])
                loss_wake = algo.get_loss_q_wake_update(mean, logvar, data[0:1,:])
                loss_sleep = algo.get_loss_q_sleep_update(mean)

                loss_q = loss_wake + loss_sleep
                self.model.zero_grad()
                loss_q.backward()

                for name, param in self.model.encoder.named_parameters():
                    if name == '2.weight' :
                        a = param.grad[0].item()
                        grads.append(a)
        elif algo_ =='iwae':
            grads = []
            for i in range(50):
                sample, mean, logvar, _, _, _ = algo.forward(data[0:1, :])
                loss = algo.get_loss(mean, logvar, data[0:1,:],)
                self.model.zero_grad()
                loss.backward()
                for name, param in self.model.encoder.named_parameters():
                    if name == '2.weight' :
                        a = param.grad[0][0].item()
                        grads.append(a)


        return grads

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
                -0.5 * torch.log(x_gh_sigma ** 2) -0.5 * (1/(x_gh_sigma ** 2)) * (data - x_gh_mean) ** 2, -1)
            log_p_h = torch.sum(h * torch.log(self.pi), -1)

            log_p_x_h = log_p_x_gh + log_p_h


            q_z_gx = torch.exp(log_q_h_gx)

            for i in range(C):
                probas = torch.eye(C)[i,:]
                h = OneHotCategorical(probas).sample((data.size()[0],))

                x_gh_sample, x_gh_mean, x_gh_sigma = self.model.decode(h)

                x_gh_logvar = torch.log(x_gh_sigma**2)
                log_p_x_gh = torch.sum(
                    -0.5 * x_gh_logvar - 0.5 * (torch.exp(-x_gh_logvar)) * (data - x_gh_mean) ** 2, -1)
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

            return diff_pi, diff_mu

        else:
            raise NotImplementedError('mode not implemented')

def visu (data,algo,exp_,algo_):
    K = [1, 5, 10, 50, 100, 1000]
    for k in K:
        grads = exp_.gradients_recog(data, algo, algo_, k)
        results = np.array(grads)
        sns.kdeplot(results, label=str(k))
    sns.plt.show()