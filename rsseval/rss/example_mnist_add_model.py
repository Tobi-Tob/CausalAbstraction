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


class ExampleMnistAddModel(nn.Module):
    """
    This model represents a handcrafted model for the MNIST addition task.
    It takes the two digit inputs D1, D2 and uses their ground truth labels (assuming the perceptual part works ideally).
    The concepts C1, C2 of each digit are artificially represented in an intermediate layer as binary encodings
    (to test and verify the DAS alignment).
    Based on C1 and C2 the final output SUM is computed.
    (Note: The extraction of the binary concept is not differentiable but is fixed.)
    """

    def __init__(self, data_split="train"):
        super(ExampleMnistAddModel, self).__init__()

        self.concept_layer = None

        # Load the dataset based on the split.
        args = SimpleNamespace(task="addition")
        if data_split == 'train':
            dataset, _, _ = load_2MNIST(args=args)
        elif data_split == 'val':
            _, dataset, _ = load_2MNIST(args=args)
        elif data_split == 'test':
            _, _, dataset = load_2MNIST(args=args)
        else:
            raise ValueError(f"Invalid split: {data_split}")

        self.data_split = data_split
        # We assume that dataset.real_concepts is a NumPy array of shape [num_samples, 2],
        # where the two columns are the ground truth digits.
        self.ground_truth = dataset.real_concepts
        self.dataset_max_index = len(self.ground_truth) - 1

    def forward(self, x):
        """
        Expects x to be either a single index or a batch of indices.
        Returns the binary concept vector (of size 8) for each sample.
        """
        self.concept_layer = self._get_binary_concepts(x)
        return self.concept_layer

    def _get_binary_concepts(self, image_index):
        """
        Expects image_index to be either a single index or a batch of indices.
        For an image index, the ground truth digit labels are retrieved and written in a stacked binary representation.
        A concept vector contains in the first 2 indices the first bit for digit label 1 (bit1_d1) and digit label 2 (bit1_d2), etc.:
        Digit label 1 and 2 ranging from 0-9 are encoded as [bit1_d1, bit1_d2, bit2_d1, bit2_d2, bit3_d1, bit3_d2, bit4_d1, bit4_d2]
        Returns a single or a batch of binary concept vectors.
        """
        # Ensure we work with a list of indices
        if isinstance(image_index, (int, np.integer)):
            indices = [image_index]
            single = True
        elif isinstance(image_index, torch.Tensor):
            # If it's a tensor, convert to a list (handle both scalars and 1D tensors)
            if image_index.dim() == 0:
                indices = [int(image_index.item())]
                single = True
            else:
                indices = image_index.tolist()
                single = False
        elif isinstance(image_index, list):
            indices = image_index
            single = len(indices) == 1
        else:
            raise ValueError("Unsupported type for image_index")

        concept_vectors = []
        for idx in indices:
            # Extract ground truth labels. Assume ground_truth is of shape [num_samples, 2].
            d1 = int(self.ground_truth[int(idx), 0])
            d2 = int(self.ground_truth[int(idx), 1])

            # Convert digits to 4-bit binary strings.
            bin1 = format(d1, '04b')  # e.g., '0001' for 1
            bin2 = format(d2, '04b')

            # Build the concept vector by interleaving bits.
            # First two entries: first bit of digit1 and digit2, etc.
            concept = []
            # for bit1, bit2 in zip(bin1, bin2):
            #     concept.append(int(bit1))
            #     concept.append(int(bit2))
            for bit1 in bin1:
                concept.append(int(bit1))
            for bit2 in bin2:
                concept.append(int(bit2))
            concept_vectors.append(concept)

        # Convert to a torch tensor.
        if single:
            return torch.tensor(concept_vectors[0], dtype=torch.float32)
        else:
            return torch.tensor(concept_vectors, dtype=torch.float32)


class PyveneWrapped(MLPForClassification):
    """
    Wraps the ExampleMnistAddModel to work with pyvene.
    Its binary concept layer is treated as the hidden state.
    """

    # Configure the MLP head. (num_classes=1 means one scalar output.)
    wrapped_config = MLPConfig(
        num_classes=1,
        n_layer=2,
        h_dim=8,
        pdrop=0.0,
        problem_type="regression",  # we intend a regression output
        squeeze_output=True,
        include_emb=False,
        include_bias=False
    )

    class FakeMLPBlock(nn.Module):
        """
        A fake block that simply returns the concept layer from the wrapped model.
        This is the hidden state that will be rotated by DAS.
        """

        def __init__(self, wrapped_model):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, x):
            # Get the stored concept layer
            h = self.wrapped_model.concept_layer
            # Ensure it has the correct shape, pyvene expects: (batch_size, 1, hidden_dim)
            if h.dim() == 2:  # (batch_size, hidden_dim)
                h = h.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, hidden_dim)
            # Detach from computation graph and prevents gradients from flowing
            # h = h.detach().requires_grad_(False)
            return h

    def __init__(self, wrapped_model: nn.Module):
        # Initialize the parent MLPForClassification with the wrapped config.
        super().__init__(self.wrapped_config)
        self.wrapped_model = wrapped_model

        # self.h is expected as argument. Replace the hidden state with our fake block that returns the concept layer.
        self.h = nn.ModuleList([self.FakeMLPBlock(self.wrapped_model)])

        # self.score is typically the linear layer that maps from h_dim to num_classes.
        self.score = nn.Linear(self.wrapped_config.h_dim, self.wrapped_config.num_classes, bias=self.wrapped_config.include_bias)

        # Set the weights of self.score to the handcrafted weight vector.
        handcrafted_weight = torch.tensor([8, 4, 2, 1, 8, 4, 2, 1], dtype=torch.float32)
        with torch.no_grad():
            self.score.weight.copy_(handcrafted_weight)
            if self.wrapped_config.include_bias:
                self.score.bias.zero_()  # or set to a fixed value as needed
        # Freeze the weights
        self.score.weight.requires_grad = False
        if self.wrapped_config.include_bias:
            self.score.bias.requires_grad = False

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
        """
        Delegates to the wrapped model to compute the concept layer.
        Then uses the handcrafted head (a differentiable linear mapping) to produce a scalar sum prediction.
        """
        if input_ids is not None:
            x = input_ids.view(-1)
            # print('base')
        elif inputs_embeds is not None:
            x = inputs_embeds.view(-1)
            # print('source')
        else:
            raise ValueError("No valid input provided (input_ids or inputs_embeds required)")
        # print('input', x)
        # Update the concept layer in the wrapped model.
        _ = self.wrapped_model(x)
        # Get the hidden state from our fake block.
        h = self.h[0](None)
        # print('h', h)
        # Apply the handcrafted linear head to compute the regression output.
        logits = self.score(h)
        # print('logits', logits)

        if return_dict:
            return SequenceClassifierOutput(loss=None, logits=logits, hidden_states=h)  # return the concept layer as hidden state
        else:
            return (logits,)


def distributed_alignment_search(target_model: nn.Module, counterfactual_data_path: str):
    """
    The two high-level variables C1 and C2 of the causal model (representing the values of a digit in a human reasoning process)
    will be encoded somewhere in a multidimensional linear subspaces of our example target model.
    In this example case we know where (in the self.h) and how (as binary bits).
    To learn these subspaces, we define an IntervenableConfig that allows us to target the concept layer, rotate or permutate it
    by an orthogonal matrix R, which allows an alignment between the high-level variables C1 and C2 with nodes in the target model.
    The loss (MSE) compares this counterfactual prediction with the ground truth sum.
    In a perfect alignment, DAS should achieve an accuracy of 1.
    """
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
    # Wrap the model
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


# Example usage:
if __name__ == '__main__':
    model = ExampleMnistAddModel(data_split="train")
    # pv_model = PyveneWrapped(model)

    # Test on a batch of indices.
    # batch_indices = [7000, 34000, 41900]
    # print(f"Input: {model.ground_truth[batch_indices]}")
    # outputs = pv_model(torch.tensor(batch_indices))
    # print(f"Output: {outputs}")

    distributed_alignment_search(target_model=model, counterfactual_data_path="data/mnist_add_counterfactual_train_data_bs100.pt")
