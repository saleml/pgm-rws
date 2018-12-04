import numpy as np
from data.utils import softmax


class GMMDataGen:
    def __init__(self, d=1, C=20, std=5.):
        """
        :param d: dimension
        :param C: number of classes
        """
        self.d = d
        self.C = C

        self.theta = np.log(6 + np.arange(C, dtype=float))
        self.latent_proba = softmax(self.theta)

        self.mus = 10 * np.arange(C, dtype=float)
        self.sigmas2 = np.repeat(std ** 2, C)

    def sampler(self, num_samples):
        for _ in range(num_samples):
            z = np.random.choice(np.arange(self.C), p=self.latent_proba)
            x = self.mus[z] + np.sqrt(self.sigmas2[z]) * np.random.randn()
            yield x

