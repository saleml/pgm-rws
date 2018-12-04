import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.nn import functional as F

ACTIVATION_FUNCTIONS = {'tanh': nn.Tanh(),
                        'relu': nn.ReLU(),
                        'sigmoid': nn.Sigmoid()}


class BasicModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, hidden_layers=2, encoding_dim=50, hidden_nonlinearity='tanh',
                 discrete=False):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.hidden_nonlinearity = hidden_nonlinearity
        self.hidden_layers = hidden_layers

        self.discrete = discrete

        if self.discrete:
            self.pre_pi = torch.zeros(encoding_dim)
            self.pi = F.softmax(self.pre_pi)

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

    def encode(self, input, reparametrize=False):
        out = self.encoder(input)
        mu = self.fc_mu(out)
        if not self.discrete:
            logsigma = self.fc_logsigma(out)
            sigma = torch.exp(logsigma)
            if not reparametrize:
                sample = Normal(mu, sigma).sample()
            else:
                eps = torch.normal(torch.zeros(mu.size()))
                sample = torch.sqrt(sigma) * eps + mu
            return sample, mu, sigma
        else:
            probas = F.softmax(mu)
            distrib = Categorical(probas)
            sample_int = distrib.sample()
            sample = torch.zeros(mu.size())
            sample[sample_int] = 1.
            return sample, probas, _

    def decode(self, sample):
        out = self.decoder(sample)
        mu = self.fc_mu_dec(out)
        if not self.discrete:
            sample = F.sigmoid(mu)
            sigma = None
        else:
            logsigma = self.fc_logsigma_dec(out)
            sigma = torch.exp(logsigma)
            sample = Normal(mu, sigma).sample()
        return sample, mu, sigma

    def forward(self, input, reparametrize=False):
        sample, mu, sigma = self.encode(input, reparametrize)
        p, mu_gen, sigma_gen = self.decode(sample)
        return (sample, mu, sigma), (p, mu_gen, sigma_gen)

    def sample(self, num_samples):
        if self.discrete:
            distrib = Categorical(self.pi)
            integer_samples = distrib.sample((num_samples, ))
            one_hot_samples = torch.zeros((num_samples, self.encoding_dim))
            one_hot_samples[torch.arange(0, num_samples), integer_samples] = 1.
            samples, _, _ = self.decode(one_hot_samples)
        else:
            z = torch.normal(torch.zeros(num_samples, self.encoding_dim))
            samples, _, _ = self.decode(z)
        return samples
