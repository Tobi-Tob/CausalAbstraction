import random
from types import SimpleNamespace
from typing import Optional
import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
import pyvene
from transformers.modeling_outputs import SequenceClassifierOutput
from task_datasets.utils.mnist_creation import load_2MNIST
from pyvene import (
    IntervenableModel,
    RotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene.models.mlp.modelings_mlp import MLPForClassification
from models.mnistdpl import MnistDPL


class WrappedMnistDPL(nn.Module):
    """
    Wraps the MnistDPL to work with pyvene.
    """
    class FakeMLPBlock(nn.Module):
        """
        A fake block that simply returns the concept layer from the wrapped model.
        This is the hidden state that will be rotated by DAS.
        """

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, cs):
            # Update the hidden state
            self.model.h = cs
            # Get the stored concept layer (possible after intervention)
            h = self.model.h  # has size (batch_size, 2, 10)
            # TODO Ensure h has the correct shape, pyvene expects: (batch_size, 1, hidden_dim)
            return h

    def __init__(self, wrapped_model: MnistDPL):
        super().__init__()
        self.wrapped_model = wrapped_model

        # Pyvene expects self.config
        self.config = SimpleNamespace(
            num_classes=19,
            n_layer=2,
            h_dim=self.wrapped_model.n_facts,
            pdrop=0.0,
            problem_type="classification",
            squeeze_output=True,
            include_emb=False,
            include_bias=False
        )

        # self.h is expected as argument. Replace the hidden state with our fake block that returns the concept layer.
        self.h = nn.ModuleList([self.FakeMLPBlock(self)])

        # config the intervention mapping with pyvene global vars
        pyvene.type_to_module_mapping[type(self)] = {
            "block_output": ("h[%s]", pyvene.models.constants.CONST_OUTPUT_HOOK),
        }
        pyvene.type_to_dimension_mapping[type(self)] = {
            "block_output": ("h_dim",),
        }

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        # Choose the input: if input_ids is provided, use it; otherwise, try inputs_embeds.
        if input_ids is not None:
            x = input_ids
        elif inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("No input provided. Please supply either input_ids or inputs_embeds.")

        # --- Reimplementation of MnistDPL to allow pyvene to intervene on self.h ---
        cs = []
        xs = torch.split(x, x.size(-1) // self.wrapped_model.n_images, dim=-1)
        for i in range(self.wrapped_model.n_images):
            lc, _, _ = self.wrapped_model.encoder(xs[i])  # sizes are ok
            cs.append(lc)
        clen = len(cs[0].shape)

        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        # We assign that to self.h so that pyvene can intervene on it.
        cs_intervened = self.h[0](cs)  # TODO maybe first apply normalize_concepts()

        pCs = self.wrapped_model.normalize_concepts(cs_intervened)
        # applies softmax on cs to ensure they represent probability distributions over the possible digit values (0-9)

        # Problog inference to compute worlds and query probability distributions
        py, _ = self.wrapped_model.problog_inference(pCs)

        logits = py
        # Return output in the expected format
        if return_dict:
            return SequenceClassifierOutput(loss=None, logits=logits, hidden_states=self.h)  # TODO is that needed?
        else:
            return (logits,)


def DAS_MnistDPL(target_model: MnistDPL, state_dict_path: str, counterfactual_data_path="data/mnist_add_counterfactual_train_data_bs100.pt"):
    """
    This method implements the Distributed Alignment Search (DAS) algorithm for MnistDPL. We want to see if the target DeepProbLog model
    implements a high-level causal abstraction model to solve the MNIST addition task.

    Are the two high-level variables C1 and C2 of the causal model (representing the concept values of a digit in a human reasoning process)
    somewhere encoded in a multidimensional linear subspaces of our target model? To answer that DAS investigates whether the behavior of the
    target model can be aligned with the data produced by the causal model. This counterfactual data is produced by changing the intermediate
    states C1 and C2 in the causal model and save its counterfactual predictions.

    For the DeepProbLog model, we focus the concept layer (after the feature extractor) by learning a rotation matrix R that permutes
    and aligns the neurons to the high-level variables C1 and C2. The loss (MSE) compares the intervention on those aligned neurons resulting
    in counterfactual predictions with the ground truth counterfactual data. In a perfect alignment, DAS should achieve an accuracy of 1.

    Args:
        target_model (MnistDPL): DeepProbLog model to be aligned
        state_dict_path (str): path to the state dictionary of the target model
        image_dataset (Dataset): dataset of the MNIST addition task to retrieve image tensors
        counterfactual_data_path (str): path to the counterfactual data (mapping of image indices to counterfactual predictions)

    Call from terminal:
    python main.py --DAS --model mnistdpl --dataset addmnist --task addition --backbone conceptizer --checkin test_model_addmnist_mnistdpl.pth --batch_size 100
    """

    target_model.load_state_dict(torch.load(state_dict_path))
    target_model.eval()
    target_model.device = "cpu"
    target_model.to(target_model.device)

    print("loading counterfactual data")
    counterfactual_dataset = torch.load(counterfactual_data_path, weights_only=True)

    # In our case our target model has “concept layer” (stored in self.h) of size 20.
    # We want to intervene on two subspaces: indices 0-3 (for C1) and indices 4-7 (for C2).
    config = IntervenableConfig(
        model_type=type(target_model),
        representations=[
            # First interventions: target the first dimensions of the concept layer.
            RepresentationConfig(
                layer=0,  # layer to target  maybe name of the layer?
                component="block_output",  # intervention type
                unit="pos",
                max_number_of_units=1,  # max number of units to align for C1
                subspace_partition=None,  # subspaces can be partitioned into continuous chunks [i, j) are the boundary indices
                # subspace_partition=[[0, 10], [10, 20]],
                intervention_link_key=0,  # Interventions can be linked to share weights and share subspaces
            ),
            # Second interventions: target the last dimensions.
            RepresentationConfig(
                layer=0,  # layer to target
                component="block_output",  # intervention type
                unit="pos",
                max_number_of_units=1,  # max number of units to align for C2
                subspace_partition=None,
                intervention_link_key=0,
            ),
        ],
        intervention_types=RotatedSpaceIntervention,
    )
    # Wrap the model to work with pyvene
    wrapped_model = WrappedMnistDPL(target_model)
    intervenable = IntervenableModel(config, wrapped_model, use_fast=True)
    # use_fast=True means the intervention location will be static within a batch.
    # For that it is important that the batch sizes of the data generation and the alignment training are the same.
    intervenable.set_device("cpu")
    intervenable.disable_model_gradients()

    # Load the dataset to retrieve image tensors
    train_dataset, _, _ = load_2MNIST(args=SimpleNamespace(task="addition"))

    # Training parameters for the rotation matrix
    epochs = 10
    lr = 0.001
    batch_size = 100
    gradient_accumulation_steps = 1

    # Optimizer: we only optimize the rotation parameters from DAS, the rest of the model is frozen.
    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
        break
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)

    def compute_metrics(eval_preds, eval_labels):
        # Assuming eval_preds are raw logits of shape [batch_size, num_classes]
        preds = torch.argmax(eval_preds, dim=1)
        correct_count = (preds == eval_labels).sum().item()
        accuracy = correct_count / eval_labels.size(0)
        return {"accuracy": accuracy}

    def compute_loss(outputs, labels):
        # If labels are provided as one-hot, you might need to convert them:
        # labels = torch.argmax(labels, dim=1)
        ce_loss = torch.nn.CrossEntropyLoss()
        return ce_loss(outputs, labels)

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
            # "input_ids": the base image tensor(s)
            # "source_input_ids": a tensor with two columns (each image tensor(s))
            # "intervention_id": an integer (0: intervene on C1 only, 1: on C2 only, 2: on both)
            # "labels": the desired class label after intervention as one-hot.
            # (We ignore base_labels here.)

            # Retrieve base images from indices.
            # Assume that batch["input_ids"] is of shape [batch_size, 1] after unsqueeze.
            base_indices = batch["input_ids"].squeeze(1)  # shape: [batch_size]
            base_images = torch.stack([train_dataset[int(idx.item())][0] for idx in base_indices])
            # Optionally, add the unsqueeze if the model expects a sequence dimension.
            base_images = base_images.unsqueeze(1)  # now shape: [batch_size, 1, C, H, W]

            # Retrieve source images from indices.
            # Assume that batch["source_input_ids"] is of shape [batch_size, 2, 1] after unsqueeze.
            source_indices = batch["source_input_ids"].squeeze(2)  # shape: [batch_size, 2]
            source_images = []
            # Loop over the two source positions (e.g., one for C1 and one for C2).
            for pos in range(source_indices.shape[1]):
                imgs = torch.stack([train_dataset[int(idx.item())][0] for idx in source_indices[:, pos]])
                source_images.append(imgs)

            # Call the intervenable model.
            if batch["intervention_id"][0] == 2:
                _, counterfactual_outputs = intervenable(
                    {"input_ids": base_images},
                    [
                        {"inputs_embeds": source_images[0]},
                        {"inputs_embeds": source_images[1]},
                    ],
                    {
                        "sources->base": (
                            [[[0]] * batch_size, [[0]] * batch_size],  # source position 0 into base position 0
                            [[[0]] * batch_size, [[0]] * batch_size],
                        )
                    },
                    subspaces=[
                        [[_ for _ in range(0, 10)]] * batch_size,
                        [[_ for _ in range(10, 20)]] * batch_size,
                    ],
                )
            elif batch["intervention_id"][0] == 0:
                _, counterfactual_outputs = intervenable(
                    {"input_ids": base_images},
                    [{"inputs_embeds": source_images[0]}, None],
                    {
                        "sources->base": (
                            [[[0]] * batch_size, None],
                            [[[0]] * batch_size, None],
                        )
                    },
                    subspaces=[
                        [[_ for _ in range(0, 10)]] * batch_size,
                        None,
                    ],
                )
            elif batch["intervention_id"][0] == 1:
                _, counterfactual_outputs = intervenable(
                    {"input_ids": base_images},
                    [None, {"inputs_embeds": source_images[0]}],
                    {
                        "sources->base": (
                            [None, [[0]] * batch_size],
                            [None, [[0]] * batch_size],
                        )
                    },
                    subspaces=[
                        None,
                        [[_ for _ in range(10, 20)]] * batch_size,
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
            loss = compute_loss(counterfactual_outputs[0], batch["labels"].squeeze().to(torch.long))

            epoch_iterator.set_postfix({"loss": loss.item(), "acc": eval_metrics["accuracy"]})

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if total_step % gradient_accumulation_steps == 0:
                optimizer.step()
                intervenable.set_zero_grad()
            total_step += 1
