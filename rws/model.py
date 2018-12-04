import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.nn import functional as F

ACTIVATION_FUNCTIONS = {'tanh': nn.Tanh(),
                        'relu': nn.ReLU(),
                        'sigmoid': nn.Sigmoid()}


class BasicModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, hidden_layers=2, encoding_dim=50, hidden_nonlinearity='tanh',
                 decoder_nonlinearity='sigmoid', discrete=False):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.hidden_nonlinearity = hidden_nonlinearity
        self.hidden_layers = hidden_layers

        self.discrete = discrete

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
        decoder_modules += [nn.Linear(hidden_dim, input_dim), ACTIVATION_FUNCTIONS[decoder_nonlinearity]]

        self.decoder = nn.Sequential(*decoder_modules)

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
            sample = distrib.sample()
            one_hot = torch.zeros(mu.size())
            one_hot[sample] = 1.
            return one_hot, probas, _

    def decode(self, sample):
        return self.decoder(sample)

    def forward(self, input, reparametrize=False):
        sample, mu, sigma = self.encode(input, reparametrize)
        p = self.decode(sample)
        return (sample, mu, sigma), p
