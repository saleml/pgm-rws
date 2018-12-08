import numpy as np
from data.utils import softmax
import torch


class GMMDataGen:
    def __init__(self, d=1, C=20, std=5., radius=10., cov_diag=None):
        """
        :param d: dimension
        :param C: number of classes
        :param std: for d=1, float standard deviation
        :param cov_diag: for d=2, 2-list for diagonal covariance
        :param radius: for d=1, distance between means, for d=2, radius of circle on which all means lie
        """
        self.d = d
        self.C = C

        # Sample the latent from a softmax distribution parametrized with [6, 7, ..., C+5]
        self.theta = np.log(1 + np.arange(C, dtype=float))
        self.latent_proba = softmax(self.theta)

        if d == 1:
            # mean = 10 * latent
            self.mus = radius * np.arange(C, dtype=float)
            # std = fixed
            self.std = std

        elif d == 2:
            if cov_diag is None:
                cov_diag = [1., 3.]
            # mean = radius [cos (2pi latent / C), sin(2pi latent / C)]
            self.mus = np.array([np.cos(2 * np.pi / C * np.arange(C, dtype=float)),
                                 np.sin(2 * np.pi / C * np.arange(C, dtype=float))]).T
            self.mus *= radius
            self.cov = np.eye(2) * cov_diag

        else:
            raise NotImplementedError("d = {} Not implemented".format(d))

    def sampler(self, num_samples):
        for _ in range(num_samples):
            z = np.random.choice(np.arange(self.C), p=self.latent_proba)
            if self.d == 1:
                x = self.mus[z] + self.std * np.random.randn()
            elif self.d == 2:
                x = np.random.multivariate_normal(self.mus[z], self.cov)
            else:
                raise NotImplementedError("d = {} Not implemented".format(self.d))
            yield x

    def next_batch(self, batch_size):
        batch = torch.zeros((batch_size, self.d))
        for i in range(batch_size):
            z = np.random.choice(np.arange(self.C), p=self.latent_proba)
            if self.d == 1:
                x = self.mus[z] + self.std * np.random.randn()
            elif self.d == 2:
                x = np.random.multivariate_normal(self.mus[z], self.cov)
            else:
                raise NotImplementedError("d = {} Not implemented".format(self.d))

            batch[i, :] = torch.from_numpy(x)
        return batch
