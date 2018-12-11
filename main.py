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
import csv

import datetime
from torch.optim.lr_scheduler import MultiStepLR


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

    parser.add_argument("--radius", type=float, default=20., help="Radius of circle containing GMM means")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mode", choices=['MNIST', 'dis-GMM', 'cont-GMM'], default='dis-GMM')
    parser.add_argument('--no-RP', action='store_true', default=False,
                        help='reparametrization trick')
    parser.add_argument("--VR", default=None, help='variance reduction')
    parser.add_argument("--d", type=int, default=2, help='dimension of data in toy gmm')
    parser.add_argument("--no-mu", action='store_true', default=False, help="learn mu in GMM")

    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")

    parser.add_argument("--milestones", nargs='*', type=int, default=[], help='scheduler milestones')
    parser.add_argument("--gamma", type=float, default=0.316, help='decay of lr at each milestone')

    parser.add_argument("--no-sleep", action='store_true', default=False, help='sleep phase of rws')
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
        train_loader = GMMDataGen(args.d, C=args.C, radius=args.radius)
        test_loader = GMMDataGen(args.d, C=args.C, radius=args.radius)
        input_dim = args.d
        encoding_dim = args.C
        model = ToyModel(input_dim, args.hidden_dim, args.hidden_layers, encoding_dim,
                         args.hidden_nonlinearity, args.mode, args.radius, not args.no_mu)
        test_set = test_loader.next_batch(1000, use_torch=False)
        test_posteriors = test_loader.get_posteriors(test_set)

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
            optimizer = Adam(list(model.parameters()) + [model.pre_pi, model.mus], lr=args.lr)
        else:
            optimizer = Adam(list(model.parameters()) + [model.pre_pi], lr=args.lr)

    optim_recog = torch.optim.Adam(encoder_params, lr=args.lr)
    optim_model = torch.optim.Adam(decoder_params, lr=args.lr)

    scheduler, scheduler_model, scheduler_recog = None, None, None
    if len(args.milestones) > 0:
        print('Using a scheduler')
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        scheduler_recog = MultiStepLR(optim_recog, milestones=args.milestones, gamma=args.gamma)
        scheduler_model = MultiStepLR(optim_model, milestones=args.milestones, gamma=args.gamma)

    # Train model
    if args.algo == 'rws':
        algo = RWS(model, optim_recog, optim_model, scheduler_recog, scheduler_model, K=args.K, mode=args.mode, sleep=not args.no_sleep)
    elif args.algo == 'vae':
        algo = Vae(model, optimizer, args.mode)
    elif args.algo == 'iwae':
        algo = IWAE(model, optimizer, scheduler, args.K, args.mode, not args.no_RP, args.VR)
    else:
        raise NotImplementedError('algo not implemented')

    time_now = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    step = 0

    path = './logs_{}/{}_{}_C{}_R{}_K{}_{}_{}'.format(args.dataset, args.algo, args.VR if args.algo == 'iwae' else ('ww' if args.no_sleep else 'ws'),
                                                  args.C, args.radius, args.K, 'nomu' if args.no_mu else 'mu', time_now)
    writer = tensorboardX.SummaryWriter(path)

    exp_ = exp(train_loader, model)

    csv_writer = csv.writer(open(path + '/log.csv', 'a', 1))
    header = ['step', 'L2_pi', 'L2_mu', 'L2_posteriors']
    if args.algo == 'rws':
        header += ['loss_model', 'loss_q_sleep', 'loss_q_wake']
    else:
        header += ['loss']
    csv_writer.writerow(header)

    if args.dataset == 'MNIST':
        for epoch in range(args.epochs):
            algo.train_loss = 0.
            for batch_idx, (data, target) in enumerate(train_loader):
                out = algo.train_step(data)
                algo.visu(writer, step, out, epoch, batch_idx, len(data), len(train_loader.dataset), args.batch_size)
                step += 1

            algo.test_log(epoch, test_loader, args.batch_size)

    if args.dataset == 'GMM':
        for step in range(100000):
            data = train_loader.next_batch(args.batch_size)
            out = algo.train_step(data)
            algo.visu(writer, step, out, latent_proba=train_loader.latent_proba, mus=train_loader.mus,
                      test_set=test_set, test_posteriors=test_posteriors, csv_writer=csv_writer, path=path)
            algo.test_log(None, test_loader, None, step)


if __name__ == "__main__":
    main()
