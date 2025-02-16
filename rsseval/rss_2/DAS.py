import argparse
import random
from types import SimpleNamespace
from typing import Optional
import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
import pyvene

from datasets.utils.mnist_creation import load_2MNIST
from pyvene import (
    IntervenableModel,
    RotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene.models.mlp.modelings_mlp import MLPForClassification
from models.mnistdpl import MnistDPL
from datasets import get_dataset
from models import get_model


class WrappedMnistDPL(nn.Module):
    """
    The WrappedMnistDPL class wraps an instance of the MnistDPL model to make it compatible with pyvene’s intervention framework.
    ConceptExtractionBlock (Nested Class):
    - Acts as a wrapper around the encoder portion of the model.
    - Since pyvene can only “hook” into nn.Module objects, reimplementing this part allows the framework to access and intervene
      on the intermediate (concept) representations.
    Pyvene Configuration:
    - The wrapper sets up a simple configuration and registers a mapping in type_to_module_mapping and type_to_dimension_mapping.
    - Pyvene knows which layer to intervene on, the target is the concept_layer output of the ConceptExtractionBlock.
    """

    class ConceptExtractionBlock(nn.Module):
        """
        A wrapper block that reimplements the behavior of the wrapped model for the concept extraction part.
        This block is needed to make the intermediate computations accessible, as pyvene can only hock onto nn.Modules.
        """

        def __init__(self, encoder, n_images):
            super().__init__()
            self.encoder = encoder
            self.n_images = n_images

        def forward(self, x):
            """
            Reimplementation of MnistDPL forward method to allow pyvene to interact with the desired concept layer (cs).
            Args:
                x: image tensor of size [bs, 1, 28, n_images x 28]

            Returns: This module outputs the hidden concept layer [bs, 1, 20] that will be rotated by DAS.
            """
            cs = []
            xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)  # splits images into "n_images" digits
            for i in range(self.n_images):
                lc, _, _ = self.encoder(xs[i])  # encodes each sub image xs into its latent representation
                cs.append(lc)
            # cs is a list containing 2x torch.Size([bs, 1, 10])
            concept_layer = torch.cat(cs, dim=-1)
            # Now, concept_layer is [bs, 1, 20]
            return concept_layer  # pyvene expects: [bs, 1, h_dim]

    def __init__(self, wrapped_model: MnistDPL):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.device = wrapped_model.device
        concept_dim = self.wrapped_model.n_facts * self.wrapped_model.n_images  # We define it as 10 * 2

        # Pyvene expects self.config
        self.config = SimpleNamespace(
            num_classes=19,
            n_layer=2,
            h_dim=concept_dim,
            pdrop=0.0,
            problem_type="classification",
            squeeze_output=True,
            include_emb=False,
            include_bias=False
        )

        # Define the layer we want pyvene to intervene on
        self.concept_layer = self.ConceptExtractionBlock(self.wrapped_model.encoder, self.wrapped_model.n_images)

        # Define the module and dimensionality for intervention mapping with pyvene global vars
        pyvene.type_to_module_mapping[type(self)] = {
            "block_output": ("concept_layer", pyvene.models.constants.CONST_OUTPUT_HOOK),
        }  # pyvene expects either a nn.Modul (e.g. "concept_layer") or a nn.ModuleList (e.g. "h[%s]")
        pyvene.type_to_dimension_mapping[type(self)] = {
            "block_output": ("h_dim",),
        }  # pyvene expects the dimensionality of the module to be: [batch_size, 1, h_dim]

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

        concept_layer = self.concept_layer(x)  # returns the potentially intervened concepts
        # Recalculate cs from concept_layer so that it is again of shape [bs, 2, 10]
        # Since concept_layer is [bs, 1, 20] and 1*20 equals 2*10, we can reshape it.
        cs = concept_layer.view(concept_layer.size(0), self.wrapped_model.n_images, self.wrapped_model.n_facts)

        pCs = self.wrapped_model.normalize_concepts(cs)
        # applies softmax on cs to ensure they represent probability distributions over the possible digit values (0-9)

        # Problog inference to compute worlds and query probability distributions
        py, _ = self.wrapped_model.problog_inference(pCs)

        logits = py
        return (logits,)


def apply_intervention(intervenable, base_images, source_images, intervention_id, batch_size):
    """
    Helper function to apply the intervention based on the intervention_id.
    Same intervention_id in the whole batch assumed!
    Args:
        intervenable: The intervenable model object.
        base_images: Tensor of base images.
        source_images: A list of tensors, one per source position.
        intervention_id: An integer (0, 1, or 2) indicating the intervention type.
        batch_size: The batch size (used for constructing mapping lists).

    Returns:
        The counterfactual outputs from the intervenable model.
    """
    if intervention_id == 2:
        # intervene on neurons aligned with C1 by overwriting with the activations from source_images[0] and C2 with source_images[1]
        _, outputs = intervenable(
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
            subspaces=[  # Define the subspace of neurons to target. We align the first half of h_dim=20 to C1 and the second half to C2
                [[i for i in range(0, 10)]] * batch_size,
                [[i for i in range(10, 20)]] * batch_size,
            ],
        )
    elif intervention_id == 0:
        # intervene only on neurons aligned with C1 by overwriting with the activations from source_images[0]
        _, outputs = intervenable(
            {"input_ids": base_images},
            [{"inputs_embeds": source_images[0]}, None],
            {
                "sources->base": (
                    [[[0]] * batch_size, None],
                    [[[0]] * batch_size, None],
                )
            },
            subspaces=[
                [[i for i in range(0, 10)]] * batch_size,
                None,
            ],
        )
    elif intervention_id == 1:
        # intervene only on neurons aligned with C2 by overwriting also with the activations from source_images[0]
        _, outputs = intervenable(
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
                [[i for i in range(10, 20)]] * batch_size,
            ],
        )
    else:
        raise ValueError("Unknown intervention_id")
    return outputs


def DAS_MnistDPL(target_model: MnistDPL, state_dict_path, counterfactual_data_path: str, args):
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
        state_dict_path: path to the state dictionary of the target model, if None, the target model is randomly initialized
        image_dataset (Dataset): dataset of the MNIST addition task to retrieve image tensors
        counterfactual_data_path (str): path to the counterfactual data (mapping of image indices to counterfactual predictions)
        args: Containing further training parameters

    Call from terminal:
    python main.py --DAS --model mnistdpl --dataset addmnist --task addition --backbone conceptizer --checkin test_model_addmnist_mnistdpl.pth --batch_size 100
    """

    # ============== Training parameters and config ==============
    epochs = args.epochs
    batch_size = args.bs
    lr = 0.001
    gradient_accumulation_steps = 1

    # In our case our target model has “concept layer” (stored in self.h) of size 20.
    # We want to intervene on two subspaces: indices 0-3 (for C1) and indices 4-7 (for C2).
    config = IntervenableConfig(
        model_type=type(target_model),
        representations=[
            # First interventions: target the first dimensions of the concept layer.
            RepresentationConfig(
                layer=0,  # layer to target
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
        intervention_types=RotatedSpaceIntervention)
    # ==============================================================

    # Load the target model
    if state_dict_path is None:
        print("Random Initialization of Target Model")
        state_dict_path = "random baseline"
    else:
        target_model.load_state_dict(torch.load(state_dict_path))
    target_model.eval()
    target_model.device = "cpu"
    target_model.to(target_model.device)

    # Load the counterfactual data
    print("loading counterfactual data")
    counterfactual_dataset = torch.load(counterfactual_data_path, weights_only=True)

    # Wrap the model to work with pyvene
    wrapped_model = WrappedMnistDPL(target_model)
    intervenable = IntervenableModel(config, wrapped_model, use_fast=True)
    # use_fast=True means the intervention location will be static within a batch.
    # For that it is important that the batch sizes of the data generation and the alignment training are the same.
    intervenable.set_device("cpu")
    intervenable.disable_model_gradients()

    # Load the dataset to retrieve image tensors
    train_dataset, _, _ = load_2MNIST(args=SimpleNamespace(task="addition"))

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
        return {"accuracy": accuracy, "correct": correct_count, "total": eval_labels.size(0)}

    def compute_loss(outputs, labels):
        # If labels are provided as one-hot, you might need to convert them:
        # labels = torch.argmax(labels, dim=1)
        ce_loss = torch.nn.CrossEntropyLoss()
        return ce_loss(outputs, labels)

    def batched_random_sampler(data):
        batch_indices = [i for i in range(int(len(data) / batch_size))]
        random.shuffle(batch_indices)
        for b_i in batch_indices:
            for i in range(b_i * batch_size, (b_i + 1) * batch_size):
                yield i

    intervenable.model.train()  # set to train mode for DAS training
    print("Distributed Intervention Training, trainable parameters: ", intervenable.count_parameters())
    train_iterator = trange(epochs, desc="Epoch")

    best_iia = 0.0  # Will store the best observed IIA (i.e., DII score)
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

            # Retrieve source images from indices.
            # Assume that batch["source_input_ids"] is of shape [batch_size, 2, 1] after unsqueeze.
            source_indices = batch["source_input_ids"].squeeze(2)  # shape: [batch_size, 2]
            source_images = []
            # Loop over the two source positions (e.g., one for C1 and one for C2).
            for pos in range(source_indices.shape[1]):
                imgs = torch.stack([train_dataset[int(idx.item())][0] for idx in source_indices[:, pos]])
                source_images.append(imgs)

            # Call the intervenable model depending on the intervention_id.
            intervention_id = batch["intervention_id"][0]
            counterfactual_outputs = apply_intervention(intervenable, base_images, source_images, intervention_id, batch_size)

            # Compute loss and metrics.
            eval_metrics = compute_metrics(counterfactual_outputs[0].detach(), batch["labels"].squeeze())

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

        """Evaluation after each epoch"""

        intervenable.model.eval()  # set model to evaluation mode
        total_correct = 0
        total_samples = 0

        eval_dataloader = DataLoader(counterfactual_dataset, batch_size=batch_size, shuffle=False)  # TODO use validation dataset
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                # Retrieve base images.
                base_indices = batch["input_ids"].squeeze(1)
                base_images = torch.stack([train_dataset[int(idx.item())][0] for idx in base_indices])

                # Retrieve source images.
                source_indices = batch["source_input_ids"].squeeze(2)
                source_images = []
                for pos in range(source_indices.shape[1]):
                    imgs = torch.stack([train_dataset[int(idx.item())][0] for idx in source_indices[:, pos]])
                    source_images.append(imgs)

                # Apply the intervention.
                intervention_id = batch["intervention_id"][0]
                counterfactual_outputs = apply_intervention(intervenable, base_images, source_images, intervention_id, batch_size)

                # Compute metrics for this batch.
                metrics = compute_metrics(counterfactual_outputs[0], batch["labels"].squeeze())
                total_correct += metrics["correct"]
                total_samples += metrics["total"]

        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch} Evaluation IIA: {epoch_accuracy:.4f}")

        # Update the best IIA (DII score) if current epoch accuracy is higher.
        if epoch_accuracy > best_iia:
            best_iia = epoch_accuracy

        # Switch back to training mode.
        intervenable.model.train()

    print(f"DII score of {state_dict_path} (highest IIA observed): {best_iia:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Alignment Search", allow_abbrev=False)
    parser.add_argument(
        "--model",
        type=str,
        default="mnistdpl",
        help="Target model to intervene on",
        choices=["mnistdpl"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="addmnist",
        help="Only MNIST addition task is supported yet",
        choices=["addmnist"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="addition",
        help="Only MNIST addition task is supported yet",
        choices=["addition"],
    )
    parser.add_argument(
        "--splitted",
        type=bool,
        default=0,
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="trained_models/mnistdpl_MNISTSingleEncoder_1.0_None.pth",
        help="Path to the state dict containing the pretrained target model; if None, the target model is randomly initialized",
    )
    parser.add_argument(
        "--joint",
        type=bool,
        default=0,
        help="Indicates whether joint the joint architecture is chosen. 0 = Single encodings of the digit images, 1 = Pair encoding",
    )
    parser.add_argument(
        "--counterfactual_data",
        type=str,
        default="data/mnist_add_counterfactual_train_data_bs100.pt",
        help="Path to the counterfactual data (mapping of image indices to counterfactual predictions) to evaluate DAS on",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=100,
        help="Batch size of the counterfactual data sampling. Its important to have the same bs used for counterfactual data generation!",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="How many epochs to train the rotation matrix",
    )
    args = parser.parse_args()

    dataset = get_dataset(args)
    encoder, decoder = dataset.get_backbone()  # get joint or disjoint architecture
    n_images, c_split = dataset.get_split()  # Based on args.joint True or False: n_images= 1 or 2 c_split= (10, 10) or (10,)
    model = get_model(args, encoder, decoder, n_images, c_split)
    print(model)
    print(args)
    DAS_MnistDPL(target_model=model, state_dict_path=args.pretrained, counterfactual_data_path=args.counterfactual_data, args=args)
