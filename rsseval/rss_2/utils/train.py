import os

import torch
import numpy as np
import wandb

from torchvision.utils import make_grid
from utils.wandb_logger import *
from utils.status import progress_bar
from datasets.utils.base_dataset import BaseDataset
from utils.dpl_loss import ADDMNIST_DPL
from utils.metrics import evaluate_metrics, evaluate_mix
from utils.generative import conditional_gen, recon_visaulization

from warmup_scheduler import GradualWarmupScheduler


def train(model, dataset: BaseDataset, _loss: ADDMNIST_DPL, args):
    """TRAINING:
        Main training loop for the MnistDPL model with dataset and loss function

        Args:
            model: network to train
            dataset (BaseDataset): dataset MNIST Addition
            _loss (ADDMNIST_DPL): loss function
            args: parsed args

        Returns:
            None: This function does not return a value.
        """
    # Default Setting for Training
    model.to(model.device)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model.opt, args.exp_decay)
    w_scheduler = GradualWarmupScheduler(model.opt, 1.0, args.warmup_steps)

    best_f1 = 0.0
    encoder_name = model.encoder.__class__.__name__
    save_path = f"{args.model}_{encoder_name}_{args.c_sup}_{args.seed}.pth"
    print("save_path", save_path)

    if args.wandb is not None:
        print('\n---wandb on\n')
        wandb.init(project=args.project, entity=args.wandb,
                   name=str(args.dataset) + "_" + str(args.model),
                   config=args)
        # wandb.config.update(args)

        # if hasattr(model, 'decoder'):
        #     list_images = make_grid(conditional_gen(model), nrow=8,)
        #     images = wandb.Image(list_images, caption="Generated samples")
        #     wandb.log({"Conditional Gen": images})

        #     list_images = make_grid(recon_visaulization(out_dict), nrow=8)
        #     images = wandb.Image(list_images, caption="Reconstructed samples")
        #     wandb.log({"Reconstruction": images})

    print('\n--- Start of Training ---\n')

    # default for warm-up
    model.opt.zero_grad()
    model.opt.step()

    for epoch in range(args.n_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            images, labels, concepts = data
            images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)

            out_dict = model(images)

            out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})

            model.opt.zero_grad()
            loss, losses = _loss(out_dict, args)

            loss.backward()
            model.opt.step()

            if args.wandb is not None:
                wandb_log_step(i, epoch, loss.item(), losses)

            if i % 10 == 0: progress_bar(i, len(train_loader) - 9, epoch, loss.item())

        model.eval()
        tloss, cacc, yacc, f1 = evaluate_metrics(model, val_loader, args)

        # update at end of the epoch 
        if epoch < args.warmup_steps:
            w_scheduler.step()
        else:
            scheduler.step()
            if hasattr(_loss, 'grade'):
                _loss.update_grade(epoch)

        ### LOGGING ###

        print('  ACC C', cacc, '  ACC Y', yacc, "F1 Y", f1)

        # Checkpointing: If the current epoch yields a higher F1 score, save the model as the best performing checkpoint.
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with F1 score: {best_f1}")

        if args.wandb is not None:
            wandb_log_epoch(epoch=epoch, acc=yacc, cacc=cacc,
                            tloss=tloss,
                            lr=float(scheduler.get_last_lr()[0]))

    # Evaluate best model on val or test
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        model.device = "cpu"
        model.to(model.device)
    if args.validate:
        print("Evaluating on validation set...")
        y_true, c_true, y_pred, c_pred = evaluate_metrics(model, val_loader, args, last=True)
    else:
        print("Evaluating on test set...")
        y_true, c_true, y_pred, c_pred = evaluate_metrics(model, test_loader, args, last=True)

    yac, yf1 = evaluate_mix(y_true, y_pred)
    cac, cf1 = evaluate_mix(c_true, c_pred)

    print([c_true, c_pred])

    print(f'Concepts:\n    ACC: {cac}, F1: {cf1}')
    print(f'Labels:\n      ACC: {yac}, F1: {yf1}')

    if args.wandb is not None:
        K = max(max(y_pred), max(y_true))

        wandb.log({'test-y-acc': yac * 100, 'test-y-f1': yf1 * 100})
        wandb.log({'test-c-acc': cac * 100, 'test-c-f1': cf1 * 100})

        wandb.log({
            'cf-labels': wandb.plot.confusion_matrix(None, y_true, y_pred, class_names=[str(i) for i in range(K + 1)]),
        })
        K = max(np.max(c_pred), np.max(c_true))
        wandb.log({
            'cf-concepts': wandb.plot.confusion_matrix(None, c_true, c_pred, class_names=[str(i) for i in range(K + 1)]),
        })

        if hasattr(model, 'decoder'):
            list_images = make_grid(conditional_gen(model), nrow=8, )
            images = wandb.Image(list_images, caption="Generated samples")
            wandb.log({"Conditional Gen": images})

            list_images = make_grid(recon_visaulization(out_dict), nrow=8)
            images = wandb.Image(list_images, caption="Reconstructed samples")
            wandb.log({"Reconstruction": images})

        wandb.finish()

    print('--- Training Finished ---')
