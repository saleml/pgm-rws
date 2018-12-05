import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical
from torch.nn import functional as F

ACTIVATION_FUNCTIONS = {'tanh': nn.Tanh(),
                        'relu': nn.ReLU(),
                        'sigmoid': nn.Sigmoid()}


class BasicModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, hidden_layers=2, encoding_dim=50, hidden_nonlinearity='tanh',
                 mode='MNIST'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.hidden_nonlinearity = hidden_nonlinearity
        self.hidden_layers = hidden_layers

        self.mode = mode

        self.pre_pi = torch.ones(encoding_dim, requires_grad=True)

        # Encoder
        transformation = ACTIVATION_FUNCTIONS[hidden_nonlinearity]
        encoder_modules = [nn.Linear(input_dim, hidden_dim), transformation]
        encoder_modules += [nn.Linear(hidden_dim, hidden_dim), transformation] * (hidden_layers - 1)

        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Linear(hidden_dim, encoding_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, encoding_dim)

        # Decoder
        decoder_modules = [nn.Linear(encoding_dim, hidden_dim), transformation]
        decoder_modules += [nn.Linear(hidden_dim, hidden_dim), transformation] * (hidden_layers - 1)

        self.decoder = nn.Sequential(*decoder_modules)

        self.fc_mu_dec = nn.Linear(hidden_dim, input_dim)
        self.fc_logsigma_dec = nn.Linear(hidden_dim, input_dim)

    @property
    def pi(self):
        return F.softmax(self.pre_pi)

    def encode(self, input, reparametrize=False):
        out = self.encoder(input)
        mu = self.fc_mu(out)
        if self.mode == 'MNIST':
            logsigma = self.fc_logsigma(out)
            sigma = torch.exp(logsigma)
            if not reparametrize:
                sample = Normal(mu, sigma).sample()
            else:
                eps = torch.normal(torch.zeros(mu.size()))
                sample = torch.sqrt(sigma) * eps + mu
            return sample, mu, sigma
        elif self.mode == 'dis-GMM':
            probas = F.softmax(mu)
            distrib = OneHotCategorical(probas)
            sample = distrib.sample()
            return sample, probas, _
        elif self.mode == 'cont-GMM':
            pass
        else:
            raise NotImplementedError('mode not implemented')

    def decode(self, sample):
        out = self.decoder(sample)
        mu = self.fc_mu_dec(out)
        if self.mode == 'MNIST':
            mu = torch.sigmoid(mu)
            sample = (mu > 0.5).float()
            sigma = torch.ones(1)  # pretty useless, just to take the log
        elif self.mode == 'dis-GMM':
            logsigma = self.fc_logsigma_dec(out)
            sigma = torch.exp(logsigma)
            sample = Normal(mu, sigma).sample()
        elif self.mode == 'cont-GMM':
            pass
        else:
            raise NotImplementedError('mode not implemented')

        return sample, mu, sigma

    def forward(self, input, reparametrize=False):
        sample, mu, sigma = self.encode(input, reparametrize)
        p, mu_gen, sigma_gen = self.decode(sample)
        return (sample, mu, sigma), (p, mu_gen, sigma_gen)

    def sample(self, num_samples):
        samples = None
        if self.mode == 'MNIST':
            distrib = OneHotCategorical(self.pi)
            z = distrib.sample((num_samples, ))
            samples, _, _ = self.decode(z)
        elif self.mode == 'dis-GMM':
            z = torch.normal(torch.zeros(num_samples, self.encoding_dim))
            samples, _, _ = self.decode(z)
        elif self.mode == 'cont-GMM':
            pass
        else:
            raise NotImplementedError('mode not implemented')
        return samples
