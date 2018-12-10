import os
import torch
from data.gmm_gen import GMMDataGen
import argparse
import matplotlib.pyplot as plt


def compare_gmm_vis(d, checkpoints, sampler):
    true_samples = sampler(1337).transpose(1,0)
    files = filter(lambda f: f[-2:] == 'pt', os.listdir(d))
    files = filter(lambda f: f.split('_')[-1][:-3] in checkpoints, files)
    files = list(files)
    samples = []
    for f in files:
        path = os.path.join(d, f)
        model = torch.load(path)
        samples.append(model.sample(1337).numpy().transpose(1,0))

    _, axx = plt.subplots(1, len(checkpoints))
    for sample, ax in zip(samples, axx):
        ax.scatter(*true_samples, label='Samples generated from true distribution')
        ax.scatter(*samples, label='Samples generated from learned model')
    plt.savefig('test.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', type=int, default=8)
    parser.add_argument('-R', type=int, default=5)
    parser.add_argument('--directory', type=str, required=True, help='Folder which contains models')
    parser.add_argument('--checkpoints', help='Checkpoints to use to load model', nargs='+')
    args = parser.parse_args()

    sampler = GMMDataGen(d=2, C=args.C, radius=args.R).next_batch
    compare_gmm_vis(args.directory, args.checkpoints, sampler)

