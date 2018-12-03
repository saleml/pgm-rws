'''Main file to be called to train a model with a given algorithm'''

from rws.model import BasicModel
from argparse import ArgumentParser
from torchvision import datasets, transforms
import torch


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
parser.add_argument("--decoder-nonlinearity", default='sigmoid',
                    help="Non linearity of the decoder")

parser.add_argument("--batch-size", type=int, default=128,
                    help="Batch size")

parser.add_argument("--dataset", default='MNIST',
                    help="Dataset to use")
args = parser.parse_args()


def main():
    # Load data
    if args.dataset == 'MNIST':
        dataset = datasets.MNIST
    train_loader = torch.utils.data.DataLoader(dataset('../data', train=True, download=True,
                                                       transform=transforms.ToTensor()),
                                               batch_size=args.batch_size, shuffle=True)

    # Transform data
    # TODO: e.g. if MNIST, make sure input is transformed to 1d
    input_dim = None  # TODO: define this

    # Create model
    model = BasicModel(input_dim, args.hidden_dim, args.hidden_layers, args.encoding_dim,
                       args.hidden_nonlinearity, args.decoder_nonlinearity)

    # Train model
    if args.algo == 'rws':
        # TODO
        pass
    elif args.algo == 'vae':
        # TODO
        pass
    elif args.algo == 'iwae':
        # TODO
        pass

    # Evaluate
    # TODO: make sure to save results (tensorboard/ csv)


if __name__ == "__main__":
    main()


