'''Main file to be called to train a model with a given algorithm'''

from rws.model import BasicModel, ToyModel
from argparse import ArgumentParser
from torchvision import datasets, transforms
import torch
from functools import partial
import tensorboardX
from rws import Vae, RWS, IWAE, exp
from torch.optim import Adam
from data.gmm_gen import GMMDataGen
import numpy as np
import matplotlib.pyplot as plt
import datetime


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--algo", default='iwae',
                        help="Algorithm to use. rws, vae, or iwae (default: rws)")

    parser.add_argument("--hidden-dim", type=int, default=200,
                        help="Number of units in hidden layers")
    parser.add_argument("--hidden-layers", type=int, default=2,
                        help="Number of hidden layers")
    parser.add_argument("--encoding-dim", type=int, default=50,
                        help="Number of units in output layer of encoder")
    parser.add_argument("--hidden-nonlinearity", default='tanh',
                        help="Non linearity of the hidden layers")

    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--K", type=int, default=5,
                        help="number of particles for IWAE and RWS")

    parser.add_argument("--dataset", default='GMM',
                        help="Dataset to use")
    parser.add_argument("--C", type=int, default=4, help="Number of GMM classes")
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--mode", choices=['MNIST', 'dis-GMM', 'cont-GMM'], default='dis-GMM')
    parser.add_argument('--no-RP', action='store_true', default=False,
                        help='reparametrization trick')
    parser.add_argument("--VR", default=None, help='variance reduction')
    parser.add_argument("--d", default=2, help='dimension of data in toy gmm')
    parser.add_argument("--no-mu", action='store_true', default=False, help="learn mu in GMM")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Load data
    if args.dataset == 'MNIST':
        transform = transforms.Compose((
            transforms.ToTensor(),
            partial(torch.flatten, start_dim=1),
            partial(torch.gt, other=0.5),
            partial(lambda x: x.float()))
        )
        dataset = datasets.MNIST('../data', train=True, download=True,
                                 transform=transform)

        input_dim = dataset[0][0].shape[1]
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = BasicModel(input_dim, args.hidden_dim, args.hidden_layers, args.encoding_dim,
                           args.hidden_nonlinearity, args.mode)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transform),
            batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'GMM':
        train_loader = GMMDataGen(args.d, C=args.C)
        test_loader = GMMDataGen(args.d, C=args.C)
        input_dim = args.d
        encoding_dim = args.C
        model = ToyModel(input_dim, args.hidden_dim, args.hidden_layers, encoding_dim,
                         args.hidden_nonlinearity, args.mode, not args.no_mu)

    else:
        raise NotImplementedError("dataset doesn't exist")

    encoder_params = list(model.encoder.parameters()) + list(model.fc_mu.parameters()) + list(
        model.fc_logvar.parameters())
    if isinstance(model, BasicModel):
        decoder_params = list(model.decoder.parameters()) + list(model.fc_mu_dec.parameters()) + list(
            model.fc_logvar_dec.parameters())
    else:
        decoder_params = []

    optimizer = Adam(model.parameters(), lr=1e-3)

    if args.mode == 'dis-GMM':
        decoder_params.append(model.pre_pi)
        if not args.no_mu:
            decoder_params.append(model.mus)
            optimizer = Adam(list(model.parameters()) + [model.pre_pi, model.mus], lr=1e-3)
        else:
            optimizer = Adam(list(model.parameters()) + [model.pre_pi], lr=1e-3)

    optim_recog = torch.optim.Adam(encoder_params, lr=1e-3)
    optim_model = torch.optim.Adam(decoder_params, lr=1e-3)

    # Train model
    if args.algo == 'rws':
        algo = RWS(model, optim_recog, optim_model, K=args.K, mode=args.mode)
    elif args.algo == 'vae':
        algo = Vae(model, optimizer, args.mode)
    elif args.algo == 'iwae':
        algo = IWAE(model, optimizer, args.K, args.mode, not args.no_RP, args.VR)
    else:
        raise NotImplementedError('algo not implemented')

    time_now = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    step = 0

    writer = tensorboardX.SummaryWriter('./logs_{}/{}_{}_C{}_K{}_{}'.format(args.dataset, args.algo, args.VR,
                                                                         args.C, args.K, time_now))

    exp_ = exp(train_loader,model)
    if args.dataset == 'MNIST':
        for epoch in range(args.epochs):
            algo.train_loss = 0.
            for batch_idx, (data, target) in enumerate(train_loader):
                out = algo.train_step(data)
                algo.visu(writer, step, out, epoch, batch_idx, len(data), len(train_loader.dataset), args.batch_size)
                step += 1

            algo.test_log(epoch, test_loader, args.batch_size)

    if args.dataset == 'GMM':
        for step in range(10000):
            data = train_loader.next_batch(args.batch_size)
            out = algo.train_step(data)
            algo.visu(writer, step, out, latent_proba=train_loader.latent_proba, mus=train_loader.mus)
            algo.test_log(None, test_loader, None, step)


if __name__ == "__main__":
    main()
