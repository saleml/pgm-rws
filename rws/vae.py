import torch.nn.functional as F
from torch.distributions import multivariate_normal
import torch


def reconstruction_loss(x, reconstruction, *args):
    return F.binary_cross_entropy(reconstruction, x, reduction='sum')


def normal_pdf_loss(x, mu, sigma, *args):
    normal = multivariate_normal(mu, sigma)
    return normal(x).sum()


class Vae(object):
    def __init__(self, model, optimizer, mode='MNIST', RP=True):
        self.model = model
        self.optimizer = optimizer
        self.mode = mode
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

        if self.mode == 'MNIST':
            return mean, logvar, loss
        elif self.mode == 'dis-GMM':
            return loss

    def test_step(self, data):
        with torch.no_grad():
            sample, mean, logvar, p_sample, p_mu, p_logvar = self.forward(data)
            loss = self.get_loss(p_mu, mean, logvar, data)

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

