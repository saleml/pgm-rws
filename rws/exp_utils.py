import numpy as np
from data import gmm_gen

class exp:
    def __init__(self, gmm_toy, model, mode):
        self.toy = gmm_toy
        self.model = model
        self.mode = mode
        self.pi = gmm_toy.latent_probas
        self.mus = gmm_toy.mus
        self.cov = gmm_toy.cov

    def gradients_recog(self, data, algo):

        return

    def gradients_model(self, data, algo):
        return

    def inference_perf(self, data):
        return

    def model_perf(self, data):
        return

