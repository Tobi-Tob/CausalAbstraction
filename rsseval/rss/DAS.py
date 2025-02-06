import random
from types import SimpleNamespace
from typing import Optional
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from torch.utils.data import Dataset, DataLoader
import pyvene
from pyvene.models.mlp.modelings_mlp import MLPConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from task_datasets.utils.mnist_creation import load_2MNIST
from pyvene import (
    IntervenableModel,
    VanillaIntervention,
    RotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene.models.mlp.modelings_mlp import MLPForClassification
from models.mnistdpl import MnistDPL


def DAS_MnistDPL(target_model: MnistDPL, state_dict_path: str, counterfactual_data_path="TODO"):
    """
    The two high-level variables C1 and C2 of the causal model (representing the values of a digit in a human reasoning process)
    will be encoded somewhere in a multidimensional linear subspaces of our example target model.
    In this example case we know where (in the self.h) and how (as binary bits).
    To learn these subspaces, we define an IntervenableConfig that allows us to target the concept layer, rotate or permutate it
    by an orthogonal matrix R, which allows an alignment between the high-level variables C1 and C2 with nodes in the target model.
    The loss (MSE) compares this counterfactual prediction with the ground truth sum.
    In a perfect alignment, DAS should achieve an accuracy of 1.

    Call from terminal:
    python main.py --DAS --model mnistdpl --dataset addmnist --task addition --backbone conceptizer --checkin test_model_addmnist_mnistdpl.pth
    """

    target_model.load_state_dict(torch.load(state_dict_path))
    target_model.eval()
    target_model.device = "cpu"
    target_model.to(target_model.device)

    print("loading counterfactual data")
    counterfactual_dataset = torch.load(counterfactual_data_path, weights_only=True)

    # In our case our target model has a single “concept layer” (stored in self.h) of size 8.
    # We want to intervene on two subspaces: indices 0-3 (for C1) and indices 4-7 (for C2).
    config = IntervenableConfig(
        model_type=type(target_model),
        representations=[
            # First interventions: target the first 4 dimensions of the concept layer.
            RepresentationConfig(
                layer=0,  # layer to target  maybe name of the layer?
                component="block_output",  # intervention type
                unit="pos",
                max_number_of_units=1,  # max number of units to align for C1
                subspace_partition=None,  # subspaces can be partitioned into continuous chunks [i, j) are the boundary indices
                # subspace_partition=[[0, 4], [4, 8]],
                intervention_link_key=0,  # Interventions can be linked to share weights and share subspaces
            ),
            # Second interventions: target the last 4 dimensions.
            RepresentationConfig(
                layer=0,  # layer to target
                component="block_output",  # intervention type
                unit="pos",
                max_number_of_units=1,  # max number of units to align for C2
                subspace_partition=None,  # maybe list(range(4, 8))  # first 4 indices
                intervention_link_key=0,
            ),
        ],
        intervention_types=RotatedSpaceIntervention,
    )
    # Wrap the model to work with pyvene
    wrapped_model = PyveneWrapped(target_model)
    intervenable = IntervenableModel(config, wrapped_model, use_fast=True)  # True means the intervention location will be static within a batch.
    intervenable.set_device("cpu")
    intervenable.disable_model_gradients()

    # Training parameters for the rotation matrix
    epochs = 10
    lr = 0.001
    batch_size = 100
    gradient_accumulation_steps = 1

    # Optimizer: we only optimize the rotation parameters from DAS.
    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
        break
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)

    def compute_metrics(eval_preds, eval_labels):
        # For regression, we check if the rounded prediction equals the target.
        total_count = 0
        correct_count = 0
        preds = torch.round(eval_preds).long()
        for pred, label in zip(preds, eval_labels):
            total_count += 1
            if pred.item() == label.item():
                correct_count += 1
        accuracy = float(correct_count) / float(total_count)
        return {"accuracy": accuracy}

    def compute_loss(outputs, labels):
        # Use Mean Squared Error loss for regression.
        mse_loss = torch.nn.MSELoss()
        return mse_loss(outputs.squeeze(), labels.float())

    def batched_random_sampler(data):
        batch_indices = [_ for _ in range(int(len(data) / batch_size))]
        random.shuffle(batch_indices)
        for b_i in batch_indices:
            for i in range(b_i * batch_size, (b_i + 1) * batch_size):
                yield i

    # We can train the rotation matrix such that we get perfect interchange intervention accuracy,
    # meaning the trained network perfectly implements the high-level algorithm on the training data.
    intervenable.model.train()  # train enables drop-off but no grads
    print("Distributed Intervention Training, trainable parameters: ", intervenable.count_parameters())
    train_iterator = trange(0, int(epochs), desc="Epoch")

    total_step = 0
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            DataLoader(
                counterfactual_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(counterfactual_dataset),
            ),
            desc=f"Epoch: {epoch}", position=0, leave=True
        )
        for batch in epoch_iterator:
            # Our counterfactual dataset is assumed to have the following keys:
            # "input_ids": the base index (or indices)
            # "source_input_ids": a tensor with two columns (each a source index)
            # "intervention_id": an integer (0: intervene on C1 only, 1: on C2 only, 2: on both)
            # "labels": the desired sum label after intervention.
            # (We ignore base_labels here.)
            batch["input_ids"] = batch["input_ids"].unsqueeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2)
            batch_size = batch["input_ids"].shape[0]
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to("cpu")

            # Call the intervenable model.
            if batch["intervention_id"][0] == 2:  # TODO check if it is ok that for the whole batch the same intervention_id is assumed
                _, counterfactual_outputs = intervenable(
                    {"input_ids": batch["input_ids"]},
                    [
                        {"inputs_embeds": batch["source_input_ids"][:, 0]},
                        {"inputs_embeds": batch["source_input_ids"][:, 1]},
                    ],
                    {
                        "sources->base": (  # TODO why is this always 0?
                            [[[0]] * batch_size, [[0]] * batch_size],  # source position 0 into base position 0
                            [[[0]] * batch_size, [[0]] * batch_size],
                        )
                    },
                    subspaces=[
                        [[_ for _ in range(0, 4)]] * batch_size,
                        [[_ for _ in range(4, 8)]] * batch_size,
                    ],
                )
            elif batch["intervention_id"][0] == 0:
                _, counterfactual_outputs = intervenable(
                    {"input_ids": batch["input_ids"]},
                    [{"inputs_embeds": batch["source_input_ids"][:, 0]}, None],
                    {
                        "sources->base": (
                            [[[0]] * batch_size, None],
                            [[[0]] * batch_size, None],
                        )
                    },
                    subspaces=[
                        [[_ for _ in range(0, 4)]] * batch_size,
                        None,
                    ],
                )
            elif batch["intervention_id"][0] == 1:
                _, counterfactual_outputs = intervenable(
                    {"input_ids": batch["input_ids"]},
                    [None, {"inputs_embeds": batch["source_input_ids"][:, 0]}],  # TODO doesnt it need to be [:, 1] here?
                    {
                        "sources->base": (
                            [None, [[0]] * batch_size],
                            [None, [[0]] * batch_size],
                        )
                    },
                    subspaces=[
                        None,
                        [[_ for _ in range(4, 8)]] * batch_size,
                    ],  # Or subspaces=1 to intervene only dimensions from 0 to 4 if defined subspace_partition=[[0, 4], [4, 8]] in config
                )
                """
                intervenable(
                    base, [None, source],
                    # 4 means token position 4
                    {"sources->base": ([None, [[4]]], [None, [[4]]])},
                    # 1 means the second partition in the config
                    subspaces=[None, [[1]]],
                    )
                """

            # Compute regression metrics.
            eval_metrics = compute_metrics(
                counterfactual_outputs[0].detach(), batch["labels"].squeeze()
            )

            # loss and backprop
            loss = compute_loss(
                counterfactual_outputs[0], batch["labels"].squeeze().to(torch.long)
            )

            epoch_iterator.set_postfix({"loss": loss.item(), "acc": eval_metrics["accuracy"]})

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if total_step % gradient_accumulation_steps == 0:
                optimizer.step()
                intervenable.set_zero_grad()
            total_step += 1
