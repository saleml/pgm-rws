'''Main file to be called to train a model with a given algorithm'''

from rws.model import BasicModel
from argparse import ArgumentParser
from torchvision import datasets, transforms
import torch
from functools import partial
from vae import Vae
from torch.optim import Adam


def parse_args():
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
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--mode", choices=['MNIST', 'dis-GMM', 'cont-GMM'], default='MNIST')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Load data
    if args.dataset == 'MNIST':
        transform = transforms.Compose((
            transforms.ToTensor(),
            partial(torch.flatten, start_dim=1),
            partial(torch.gt, other=0.5))
        )
        dataset = datasets.MNIST('../data', train=True, download=True,
                                 transform=transform)
        input_dim = dataset[0][0].shape[1]
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size, shuffle=True)

    # Create model
    model = BasicModel(input_dim, args.hidden_dim, args.hidden_layers, args.encoding_dim,
                       args.hidden_nonlinearity)
    optimizer = Adam(model.parameters())

    # Train model
    if args.algo == 'rws':
        # TODO
        pass
    elif args.algo == 'vae':
        vae = Vae(model, optimizer, args.mode)
        train_step = vae.train_step
    elif args.algo == 'iwae':
        # TODO
        pass

    for i in range(args.epochs):
        print('Epoch %s' %i)
        for batch in train_loader:
            data = batch[0].type(torch.FloatTensor)
            loss = train_step(data)

    # Evaluate
    # TODO: make sure to save results (tensorboard/ csv)


if __name__ == "__main__":
    main()


