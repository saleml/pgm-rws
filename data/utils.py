import numpy as np
from itertools import permutations


def softmax(z):
    return np.exp(z)/(np.sum(np.exp(z)))


def L2_difference_perm(mu, predicted, C):
    perms = permutations(range(C))
    best_perm = None
    best_L2 = None
    for perm in perms:
        pred = predicted[list(perm)]
        L2 = np.sqrt(np.sum((mu - pred) ** 2))
        if best_L2 is None or L2 < best_L2:
            best_L2 = L2
            best_perm = perm
    return best_L2 / C, list(best_perm)