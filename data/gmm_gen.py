import numpy as np
from data.utils import softmax
from scipy.stats import multivariate_normal
import torch


class GMMDataGen:
    def __init__(self, d=1, C=20, radius=20., cov_diag=None):
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
            self.mus = np.arange(C, dtype=float)
            cov_diag = [[1.]]

        elif d == 2:
            if cov_diag is None:
                cov_diag = [1., 3.]
            # mean = radius [cos (2pi latent / C), sin(2pi latent / C)]
            self.mus = np.array([np.cos(2 * np.pi / C * np.arange(C, dtype=float)),
                                 np.sin(2 * np.pi / C * np.arange(C, dtype=float))]).T

        else:
            raise NotImplementedError("d = {} Not implemented".format(d))

        self.mus *= radius
        self.cov = np.eye(d) * cov_diag

    def sampler(self, num_samples):
        for _ in range(num_samples):
            z = np.random.choice(np.arange(self.C), p=self.latent_proba)
            x = np.random.multivariate_normal(self.mus[z], self.cov)
            yield x

    def next_batch(self, batch_size, use_torch=True):
        batch = torch.zeros((batch_size, self.d))
        for i in range(batch_size):
            z = np.random.choice(np.arange(self.C), p=self.latent_proba)
            x = np.random.multivariate_normal(self.mus[z], self.cov)
            if use_torch:
                batch[i, :] = torch.from_numpy(x)
        return batch

    def get_posteriors(self, test_samples):
        m = test_samples.shape[0]
        post = np.zeros((m, self.C))
        for z in range(self.C):
            post[:, z] = multivariate_normal.pdf(test_samples, self.mus[z], self.cov)
        post *= self.latent_proba
        total_proba = np.sum(post, axis=1)
        return (post.T / total_proba).T

