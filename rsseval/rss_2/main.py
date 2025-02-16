# ================================
# This is the main module
# This module initializes the environment and model, parses arguments,
# and controls the overall flow for training and testing.
# ================================

import argparse
import datetime
import importlib
import sys

import setproctitle
import socket
import uuid

from datasets import get_dataset
from models import get_model
from utils.args import *
from utils.checkpoint import create_load_ckpt
from utils.conf import *
from utils.train import train

conf_path = os.getcwd() + "."
sys.path.append(conf_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Reasoning Shortcut', allow_abbrev=False)
    parser.add_argument('--model', type=str, default='cext', help='Model for inference.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true', help='Loads the best arguments for each method, '
                                                                      'dataset and memory buffer.')

    torch.set_num_threads(4)

    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    # LOAD THE PARSER SPECIFIC OF THE MODEL, WITH ITS SPECIFICS
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    parser.add_argument('--project', type=str, default="Reasoning-Shortcuts", help='wandb project')

    args = parser.parse_args()  # this is the return

    # load args related to seed etc.
    set_random_seed(args.seed) if args.seed is not None else set_random_seed(42)

    return args


# Main function: Orchestrates data loading, model initialization, training/testing,
# and post-training evaluation or probing based on the parsed arguments.
def main(args):
    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    # Load dataset, model, loss, and optimizer
    encoder, decoder = dataset.get_backbone()
    n_images, c_split = dataset.get_split()
    model = get_model(args, encoder, decoder, n_images, c_split)
    print(model)
    loss = model.get_loss(args)
    model.start_optim(args)

    # SAVE A BASE MODEL OR LOAD IT, LOAD A CHECKPOINT IF PROVIDED
    model = create_load_ckpt(model, args)

    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    # perform posthoc evaluation/ cl training/ joint training
    print('    Chosen device:', model.device)
    if args.posthoc:
        pass
    else:
        train(model, dataset, loss, args)

    print('\n ### Closing ###')


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)

"""
Example call:
python main.py --model mnistdpl --dataset addmnist --task addition --n_epochs 20 --c_sup 0 --seed 2 --validate --wandb tobi-tob-tu-darmstadt


General Arguments

    --model
        Description: Specifies the model for inference or training.
        Choices: {boiacbm, boiadpl, boialtn, boiann, cext, cvae, kandcbm, kandclip, kanddpl, kandltn, kandnn, kandpreprocess, minikanddpl, mnistcbm,
        mnistclip, mnistdpl, mnistdplrec, mnistltn, mnistnn, mnistpcbmdpl, mnistpcbmltn, mnistpcbmsl, mnistsl, mnistslrec, mnmathcbm, mnmathdpl,
        mnmathnn, presddoiadpl, sddoiacbm, sddoiaclip, sddoiadpl, sddoialtn, sddoiann, xorcbm, xordpl, xornn}
        Example: --model mnistdpl

    --dataset
        Description: The dataset to be used for training or testing.
        Choices:
        shortmnist (Shortcut MNIST ? A simplified version of the MNIST dataset, focused on learning binary classification tasks like even vs. odd)
        clipshortmnist (A variation of shortmnist where features are preprocessed or extracted using CLIP - Contrastive Language-Image Pretraining)
        halfmnist (???)
        mnmath (Likely a dataset involving mathematical tasks, such as addition or subtraction using MNIST digits)
        restrictedmnist (???)
        xor (A dataset for learning the XOR logical operation, often used as a toy problem in explainability research)

    --joint
        Description: A flag that indicates whether joint training is enabled. Uses a fully connected NN to process the digits together.
        If True, MNISTPairsEncoder is used else MNISTSingleEncoder 

    --load_best_args
        Description: Loads predefined optimal arguments for the method, dataset, and memory buffer.

    --seed
        Description: Sets the random seed for reproducibility.

    --notes
        Description: Adds notes for this run.
        Example: --notes "Testing new architecture"

    --non_verbose
        Description: Suppresses detailed logging output.

    --wandb
        Description: Enables logging to WandB and sets the project name. If you used WandB for logging during training, evaluation metrics and
        visualizations will appear on the WandB dashboard.
        Example: --wandb my_project

    --checkin
        Description: Specifies the path to load a checkpoint.
        Example: --checkin /path/to/checkpoint

    --checkout
        Description: Saves the trained model to data/ckpts.

    --posthoc
        Description: The --posthoc flag loads the saved model and evaluates it directly on the test dataset. This bypasses any training process and
        focuses solely on performance metrics. Uses test.py script that searches for saved models according to the provided args, also seed is fixed
        Example: python main.py --dataset shortmnist --model mnistdpl --posthoc

    --probe
        Description: The --probe flag performs a non-linear probing analysis of the trained model. It evaluates how well the model's learned
        representations can perform on secondary tasks or metrics.
        Example: python main.py --dataset shortmnist --model mnistdpl --checkin best_model_shortmnist_mnistdpl_0.pth --probe

    --validate
        Description: If you are tuning hyperparameters, you can use --validate to evaluate the model on the validation set. This is used to assess
        performance during optimization.
        Example: python main.py --dataset shortmnist --model mnistdpl --checkin best_model_shortmnist_mnistdpl_0.pth --validate

    --preprocess
        Description: Preprocesses the dataset.
        Datatype: flag (no value required)
        Example: --preprocess

Training Arguments

    --n_epochs
        Description: Number of training epochs.
        Example: --n_epochs 20

    --lr
        Description: Learning rate for optimization.
        Example: --lr 0.001

    --batch_size
        Description: Number of samples per training batch.
        Example: --batch_size 64

    --exp_decay
        Description: Exponential decay factor for learning rate.
        Example: --exp_decay 0.9

    --c_sup
        Description: The --c_sup argument controls the supervision coefficient used during training. It determines how much weight is assigned to the
        supervised loss term.
        Example: --c_sup 0.05

    --task
        Description: Specifies the task, such as addition.
        Potential Choices: addition, subtraction, classification, logic

    --backbone
        Description: Specifies the model's backbone (e.g., conceptizer or neural).
        Example: --backbone conceptizer
"""

