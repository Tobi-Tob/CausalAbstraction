# ================================
# This is the main module
# This module initializes the environment, parses arguments,
# and controls the overall flow for training, testing, or tuning.
# ================================

import sys, os
import torch
import argparse
import importlib
import setproctitle, socket, uuid
import datetime

from task_datasets import get_dataset
from models import get_model
from utils.train import train
from utils.test import test
from utils.preprocess_resnet import preprocess
from utils.conf import *
import signal
from utils.args import *
from utils.checkpoint import save_model, create_load_ckpt
from utils.probe import probe

from argparse import Namespace
import wandb

conf_path = os.getcwd() + "."
sys.path.append(conf_path)


# Define a custom exception to handle external termination signals (e.g., SIGINT, SIGTERM).
class TerminationError(Exception):
    """Error raised when a termination signal is received"""

    def __init__(self):
        """Init method

        Args:
            self: instance

        Returns:
            None: This function does not return a value.
        """
        super().__init__("External signal received: forcing termination")


# Signal handler: Raises TerminationError upon receiving termination signals.
def __handle_signal(signum: int, frame):
    """For program termination on cluster

    Args:
        signum (int): signal number
        frame: frame

    Returns:
        None: This function does not return a value.

    Raises:
        TerminationError: Always.
    """
    raise TerminationError()


# Setup function to register signal handlers for graceful termination.
def register_termination_handlers():
    """Makes this process catch SIGINT and SIGTERM. When the process receives such a signal after this call, a TerminationError is raised.

    Returns:
        None: This function does not return a value.
    """

    signal.signal(signal.SIGINT, __handle_signal)
    signal.signal(signal.SIGTERM, __handle_signal)


# Parse command-line arguments and perform initial configuration (e.g., random seed setup).
def parse_args():
    """Parse command line arguments

    Returns:
        args: parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Reasoning Shortcut", allow_abbrev=False
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cext",
        help="Model for inference.",
        choices=get_all_models(),
    )
    parser.add_argument(
        "--load_best_args",
        action="store_true",
        help="Loads the best arguments for each method, " "dataset and memory buffer.",
    )

    torch.set_num_threads(4)

    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module("models." + args.model)

    # LOAD THE PARSER SPECIFIC OF THE MODEL, WITH ITS SPECIFICS
    get_parser = getattr(mod, "get_parser")
    parser = get_parser()
    parser.add_argument(
        "--project", type=str, default="Reasoning-Shortcuts", help="wandb project"
    )
    add_test_args(parser)
    args = parser.parse_args()  # this is the return

    # load args related to seed etc.
    set_random_seed(args.seed) if args.seed is not None else set_random_seed(42)

    return args


# Function for hyperparameter tuning using WandB sweeps.
# Configures the search space and launches training runs for each sweep iteration.
def tune(args):
    """
    This function performs a hyper-parameter tuning of the model using a WandB sweep.

    Args:
        args: parsed command line arguments
    """
    sweep_conf = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": args.val_metric},
        "parameters": {
            "batch_size": {"values": [32, 64, 128, 256, 512]},
            "lr": {"values": [0.0001, 0.001, 0.01]},
            "weight_decay": {"values": [0.0, 0.0001, 0.001, 0.01, 0.1]},
        },
    }

    if "ltn" in args.model:
        sweep_conf["parameters"]["p"] = {"values": [2, 4, 6, 8, 10]}
        sweep_conf["parameters"]["and_op"] = {"values": ["Godel", "Prod"]}
        sweep_conf["parameters"]["or_op"] = {"values": ["Godel", "Prod"]}
        sweep_conf["parameters"]["imp_op"] = {"values": ["Godel", "Prod"]}

    if args.c_sup > 0:
        sweep_conf["parameters"]["w_c"] = {"values": [1, 2, 5]}

    if args.entropy > 0:
        sweep_conf["parameters"]["w_h"] = {"values": [1, 2, 5, 8, 10]}

    def train_conf():
        with wandb.init(project=args.proj_name, config=sweep_conf, entity=args.entity):
            config = wandb.config
            args.batch_size = config.batch_size
            args.lr = config.lr
            args.weight_decay = config.weight_decay
            if "ltn" in args.model:
                args.p = config.p
                args.and_op = config.and_op
                args.or_op = config.or_op
                args.imp_op = config.imp_op
            dataset = get_dataset(args)

            # Load dataset, model, loss, and optimizer
            encoder, decoder = dataset.get_backbone()
            n_images, c_split = dataset.get_split()
            model = get_model(args, encoder, decoder, n_images, c_split)
            loss = model.get_loss(args)
            model.start_optim(args)

            train(model, dataset, loss, args)

    sweep_id = wandb.sweep(sweep=sweep_conf, project=args.proj_name)
    wandb.agent(sweep_id, function=train_conf, count=args.count)


# Main function: Orchestrates data loading, model initialization, training/testing,
# and post-training evaluation or probing based on the parsed arguments.
def main(args):
    """Main function. Provides functionalities for training, testing and active learning.

    Args:
        args: parsed command line arguments.

    Returns:
        None: This function does not return a value.
    """
    if not args.tuning:
        # Add uuid, timestamp and hostname for logging
        args.conf_jobnum = str(uuid.uuid4())
        args.conf_timestamp = str(datetime.datetime.now())
        args.conf_host = socket.gethostname()
        dataset = get_dataset(args)

        # Load dataset, model, loss, and optimizer
        encoder, decoder = dataset.get_backbone()  # TL: joint or disjoint
        n_images, c_split = dataset.get_split()  # Based on args.joint True or False: n_images= 1 or 2 c_split= (10, 10) or (10,)
        model = get_model(args, encoder, decoder, n_images, c_split)
        loss = model.get_loss(args)
        model.start_optim(args)

        # SAVE A BASE MODEL OR LOAD IT, LOAD A CHECKPOINT IF PROVIDED
        # model = create_load_ckpt(model, args)

        # set job name
        setproctitle.setproctitle(
            "{}_{}_{}".format(
                args.model,
                args.buffer_size if "buffer_size" in args else 0,
                args.dataset,
            )
        )

        # perform posthoc evaluation/ cl training/ joint training
        print("    Chosen device:", model.device)

        if args.preprocess:  # TODO what does this do?
            preprocess(model, dataset, args)
            print("\n ### Closing ###")
            quit()

        if args.probe:
            probe(model, dataset, args)
        elif args.posthoc:
            test(model, dataset, args)  # test the model if post-hoc is passed

        elif args.DAS:  # Apply Distributed Alignment Search
            if args.model == "mnistdpl":
                from DAS import DAS_MnistDPL
                DAS_MnistDPL(target_model=model, state_dict_path=args.checkin)
            else:
                raise NotImplementedError(f'Distributed Alignment Search (DAS) is not implemented for {args.model}')

        else:
            train(model, dataset, loss, args)  # train the model otherwise
            save_model(model, args)  # save the model parameters
    else:
        tune(args)

    print("\n ### Closing ###")


if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
Example call:
python main.py --model mnistdpl --dataset addmnist --task addition --backbone conceptizer --n_epochs 2 --validate --wandb tobi-tob-tu-darmstadt

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

    --load_best_args
        Description: Loads predefined optimal arguments for the method, dataset, and memory buffer.

    --seed
        Description: Sets the random seed for reproducibility.
        Example: --seed 42

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
        Example: --checkout

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
    
    --DAS
        Description: Run Distributed Alignment Search (DAS)
        Datatype: flag (no value required)

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

Hyperparameter Tuning Arguments

    --tuning
        Description: The --tuning argument enables hyperparameter tuning for the model using a Bayesian search strategy. This allows for automated optimization of key hyperparameters to maximize or minimize a given validation metric (e.g., F1 score, accuracy, etc.). The script runs a search process over a predefined range of hyperparameters (e.g., learning rate, batch size, weight decay) to find the best combination.
        Example: python main.py --dataset shortmnist --model mnistdpl --n_epochs 20 --lr 0.001 --batch_size 64 --exp_decay 0.99 --c_sup 0 --checkout --task addition --proj_name MNIST-DPL --tuning --val_metric f1

    --val_metric
        Description: The --val_metric argument specifies the metric that the tuning process will optimize. This metric is calculated on the validation set during tuning runs.
        Choices: --val_metric {f1, accuracy, precision, recall, loss}

    --proj_name
        Description: Project name used during WandB tuning.
        Example: --proj_name MNIST-DPL
"""
