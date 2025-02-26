# Fully neural model for MNIST
import torch
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import ADDMNIST_DPL


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'Concept Extractor .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistNN(nn.Module):
    NAME = 'mnistnn'
    """Fully neural MODEL FOR MNIST"""

    def __init__(self, encoder, n_images=2,
                 c_split=(), args=None,
                 n_facts=20):
        """Initialize method

                Args:
                    self: instance
                    encoder (nn.Module): encoder
                    n_images (int, default=2): number of images
                    c_split: concept splits
                    args: command line arguments
                    model_dict (default=None): model dictionary
                    n_facts (int, default=20): number of concepts
                    nr_classes (int, nr_classes): number of classes

                Returns:
                    None: This function does not return a value.
                """
        super(MnistNN, self).__init__()

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split
        self.encoder = encoder
        self.joint = args.joint
        self.h_dim = 20  # Concept layer

        if args.task == "addition":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 19
        elif args.task == "product":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 37
        elif args.task == "multiop":
            self.n_facts = 5
            self.nr_classes = 3

        self.classifier = nn.Sequential(
            nn.Flatten(),  # This will convert [bs, 1, 20] to [bs, 20]
            nn.Linear(self.h_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.nr_classes),
            nn.Softmax(dim=1),
        )

        # opt and device
        self.opt = None
        self.device = get_device()

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dict
        """
        # Image encoding for combined (PairsEncoder) or separate (SingleEncoder) images
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, mu, logvar = self.encoder(xs[i])  # sizes are ok
            # lc: Latent concept representation extracted from the image [bs, 2/n_images, n_facts]
            cs.append(lc)
        concept_layer = torch.cat(cs, dim=1).view(x.size(0), 1, -1)  # final shape [bs, 1, 20]

        # Classification head
        py = self.classifier(concept_layer)  # output probabilities for the sums
        # Add a small offset against numerical instabilities
        py = py + 1e-5
        with torch.no_grad():
            Z = torch.sum(py, dim=-1, keepdim=True)
        py = py / Z

        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)  # [bs, 2, n_facts]

        return {'CS': cs, 'YS': py}

    @staticmethod
    def get_loss(args):
        if args.dataset in ['addmnist', 'shortmnist', 'restrictedmnist']:
            return ADDMNIST_DPL(ADDMNIST_Cumulative)
        else:
            return NotImplementedError('Wrong dataset choice')

    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr, weight_decay=args.weight_decay)
