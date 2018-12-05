'''Main file to be called to train a model with a given algorithm'''

from rws.model import BasicModel
from rws.algos import RWS
from argparse import ArgumentParser
from torchvision import datasets, transforms
import torch
from functools import partial
import tensorboardX


# Parse arguments

parser = ArgumentParser()
parser.add_argument("--algo", default='rws',
                    help="Algorithm to use. rws, vae, or iwae (default: rws)")
parser.add_argument("--variance-reduction",
                    help="Variance reduction technique for inference network gradients var. reduction (default:None)")

parser.add_argument("--model", default='basic',
                    help="Architecture to use. basic or double (default: basic)")
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

parser.add_argument("--dataset", default='MNIST',
                    help="Dataset to use")

parser.add_argument("--K", default=5,
                    help="number of particles for IWAE and RWS")

parser.add_argument("--mode", default='MNIST',
                    help="MNIST, dis-GMM, cont-GMM mode ")

parser.add_argument("--epochs", default=50,
                    help="number of training epochs ")
args = parser.parse_args()


def main():
    # Load data
    if args.dataset == 'MNIST':
        transform = transforms.Compose((
            transforms.ToTensor(),
            partial(torch.flatten, start_dim=1),
            partial(torch.gt, other=0.5),
            partial(lambda x : x.float()))

        )
        dataset = datasets.MNIST('../data', train=True, download=True,
                                 transform=transform)
        input_dim = dataset[0][0].shape[1]
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size, shuffle=True)

    # Create model
    model = BasicModel(input_dim, args.hidden_dim, args.hidden_layers, args.encoding_dim,
                       args.hidden_nonlinearity)

    encoder_params = list(model.encoder.parameters()) + list(model.fc_mu.parameters()) + list(model.fc_logsigma.parameters())
    decoder_params = list(model.decoder.parameters()) + list(model.fc_mu_dec.parameters()) + list(model.fc_logsigma_dec.parameters())

    if model.discrete :
        decoder_params += list(model.pre_pi)


    optim_recog = torch.optim.Adam(encoder_params, lr = 1e-3)
    optim_model = torch.optim.Adam(decoder_params, lr = 1e-3)

    # Train model
    if args.algo == 'rws':

        algo = RWS(model,optim_recog, optim_model, K = args.K)

    elif args.algo == 'vae':
       update = vae.update
    elif args.algo == 'iwae':
        # TODO
        pass

    writer = tensorboardX.SummaryWriter('/Users/assouel/PycharmProjects/pgm-rws/logs/')
    step = 0
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):

            args = algo.train_step(data)

            if (batch_idx%10) ==0 :
                algo.visu(writer, step, args)
            step +=1

    # Evaluate
    # TODO: make sure to save results (tensorboard/ csv)


if __name__ == "__main__":
    main()


