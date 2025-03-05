import argparse
import csv
import os
import random
import re
import shutil
from types import SimpleNamespace
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from datasets.utils.mnist_creation import load_2MNIST
from pyvene import (
    IntervenableModel,
    RotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from datasets import get_dataset
from models import get_model
from wrapped_models import WrappedMnistDPL, WrappedMnistNN


class AlignmentHypothesis:
    def __init__(self, i, h_dim=20):
        """
        This class encapsulates the subspace indices based on a given i.
        It defines the alignment of a hidden layer with the causal abstraction variables C1 and C2.
          - C1: dimensions [0, i)
          - C2: dimensions [i, 2*i)
          - None: dimensions [2*i, h_dim)
        """
        self.id = i
        self.h_dim = h_dim
        self.C1_dims = list(range(0, i))
        self.C2_dims = list(range(i, 2 * i))
        self.none_dims = list(range(2 * i, h_dim))
        self.description = f"C1: {self.C1_dims}, C2: {self.C2_dims}, None: {self.none_dims}"


def apply_intervention(intervenable, base_images, source_images, intervention_id, batch_size, alignment_hypothesis):
    """
    Helper function to apply the intervention based on the intervention_id.
    Same intervention_id in the whole batch assumed!
    Args:
        intervenable: The intervenable model object.
        base_images: Tensor of base images.
        source_images: A list of tensors, one per source position.
        intervention_id: An integer (0, 1, or 2) indicating the intervention type.
        batch_size: The batch size (used for constructing mapping lists).
        alignment_hypothesis: AlignmentHypothesis object that defines the subspaces to intervene on.
    Returns:
        The counterfactual outputs from the intervenable model.
    """
    C1_dims = alignment_hypothesis.C1_dims
    C2_dims = alignment_hypothesis.C2_dims
    assert intervenable.model.h_dim == alignment_hypothesis.h_dim  # assumed to be 20
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
            subspaces=[  # Define the subspace of neurons to target using the indices given by the hypothesis
                [C1_dims] * batch_size,
                [C2_dims] * batch_size,
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
                [C1_dims] * batch_size,
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
                [C2_dims] * batch_size,
            ],
        )
    else:
        raise ValueError("Unknown intervention_id")
    return outputs


def init_intervenable(model_to_wrap):
    """
    This helper function defines the IntervenableConfig and returns the pyvene intervenable model.
    """
    # Wrap the model to work with pyvene
    if model_to_wrap.__class__.__name__ == "MnistNN":
        wrapped_model = WrappedMnistNN(model_to_wrap)
    elif model_to_wrap.__class__.__name__ == "MnistDPL":
        wrapped_model = WrappedMnistDPL(model_to_wrap)
    else:
        raise TypeError(f"No wrapper model implemented for {model_to_wrap.__class__.__name__}")

    # In our case our target model has “concept layer” (stored in self.h) of size 20.
    # We want to intervene on two subspaces: indices 0-3 (for C1) and indices 4-7 (for C2).
    config = IntervenableConfig(
        model_type=type(wrapped_model),
        representations=[
            # First interventions: target the first dimensions of the concept layer.
            RepresentationConfig(
                layer=0,  # layer to target
                component="block_output",  # intervention type
                unit="pos",
                max_number_of_units=1,  # max number of units to align for C1
                subspace_partition=None,  # subspaces can be partitioned into continuous chunks [i, j) are the boundary indices
                # subspace_partition=[[0, 10], [10, 20]], is already handled when calling the intervenable
                intervention_link_key=0,  # Interventions can be linked to share weights and share subspaces
            ),
            # Second interventions: target the last dimensions.
            RepresentationConfig(
                layer=0,
                component="block_output",
                unit="pos",
                max_number_of_units=1,  # max number of units to align for C2
                subspace_partition=None,
                intervention_link_key=0,
            ),
        ],
        intervention_types=RotatedSpaceIntervention)

    intervenable = IntervenableModel(config, wrapped_model, use_fast=True)
    # use_fast=True means the intervention location will be static within a batch.
    # For that it is important that the batch sizes of the data generation and the alignment training are the same.
    intervenable.set_device("cpu")
    intervenable.disable_model_gradients()
    return intervenable


def distributed_alignment_search(target_model, state_dict_path, alignment_hypothesis=AlignmentHypothesis(10), save_dir=None):
    """
    This method implements the Distributed Alignment Search (DAS) algorithm for MnistDPL or MnistNN. We want to see if the target model
    implements a high-level causal abstraction model to solve the MNIST addition task.

    Are the two high-level variables C1 and C2 of the causal model (representing the concept values of a digit in a human reasoning process)
    somewhere encoded in a multidimensional linear subspaces of our target model? To answer that DAS investigates whether the behavior of the
    target model can be aligned with the data produced by the causal model. This counterfactual data is produced by changing the intermediate
    states C1 and C2 in the causal model and save its counterfactual predictions.

    For the DeepProbLog and purely neural model, we focus the concept layer (after the feature extractor) by learning a rotation matrix R that
    permutes and aligns the neuron activations to the high-level variables C1 and C2. We intervene on those rotated activations and rotate it back.
    The loss compares the intervention on those aligned neurons resulting in counterfactual predictions with the ground truth counterfactual data.
    In a perfect alignment, DAS should achieve an accuracy of 1 (assuming the target model can solve the task 100%).

    Args:
        target_model: MnistDPL or MnistNN model to be aligned with the causal abstraction
        state_dict_path: path to the state dictionary of the target model, if None, the target model is randomly initialized
        alignment_hypothesis: AlignmentHypothesis object that defines the subspaces to intervene on. Default is C1 in index [0,9] and C2 in [10,19]
        save_dir: Optional argument where to save the matrix R, if non it is saved next to the model state_dict

    Call from terminal:
    python main.py --DAS --model mnistdpl --dataset addmnist --task addition --backbone conceptizer --checkin test_model_addmnist_mnistdpl.pth --batch_size 100
    """

    # ============== Training parameters ===========================
    epochs = 8
    batch_size = 100  # Set this according to the bs in the counterfactual data!
    lr = 0.02
    lr_decay = 0.80
    binary_reg_weight = 0.005  # Push every entry in the rotation matrix to be binary (0.0 or 1.0)
    permutation_reg_weight = 0.0  # Removed... Push the matrix to solutions solution with less inner block permutation (off-diagonals are 0)
    gradient_accumulation_steps = 1
    # ==============================================================

    # Load the target model
    if state_dict_path is None:
        print("Random Initialization of Target Model")
        state_dict_path = "random_baseline"
    else:
        target_model.load_state_dict(torch.load(state_dict_path))
    target_model.eval()
    target_model.device = "cpu"
    target_model.to(target_model.device)

    # Build the intervenable (pyvene) model
    intervenable = init_intervenable(target_model)

    # Optimizer: we only optimize the rotation parameters from DAS, the rest of the model is frozen.
    optimizer_params = [{"params": next(iter(intervenable.interventions.values()))[0].rotate_layer.parameters()}]
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # Check if batch size is compatible with counterfactual dataset batch size
    if "bs" + str(batch_size) not in counterfactual_train:
        raise Warning(f"DAS training bs is '{batch_size}', make sure it is the same batch size that is used for counterfactual data generation.")

    def compute_metrics(eval_preds, eval_labels):
        # Assuming eval_preds are raw logits of shape [batch_size, num_classes]
        preds = torch.argmax(eval_preds, dim=1)
        correct_count = (preds == eval_labels).sum().item()
        accuracy = correct_count / eval_labels.size(0)
        return {"accuracy": accuracy, "correct": correct_count, "total": eval_labels.size(0)}

    def compute_loss(outputs, labels, rotation, binary_reg_weight, permutation_reg_weight):
        """
        Computes the total loss as a combination of a data-dependent loss (Cross Entropy in this case) and two regularization terms:

        1. Data Loss (Cross Entropy):
            Measures the discrepancy between the counterfactual predictions and the ground truth labels.

        2. Binary Regularization:
            Encourages every element of the rotation matrix `R` to be near binary values (0 or 1).
            The regularization term is defined as:
            L_binary = sum((R_ij * (1 - R_ij))^2)
            This term is zero when all R_ij are exactly 0 or 1 and positive otherwise.

        3. Permutation Regularization:
            Enforces an identity-like (or block-diagonal) structure on the rotation matrix by penalizing off-diagonal values
            in each of its sub-blocks. Goal of the regularization is to discourage "inner block permutations" within each sub block of the matrix.
            The matrix R is partitioned into blocks: R11, R12, R21, and R22.
            For each block, the penalty is:
            L_perm_block = || block - diag(block) ||_F

        The overall loss is then:
            L_total = L_ce + binary_reg_weight * L_binary + permutation_reg_weight * L_perm

        Args:
            outputs (Tensor): Counterfactual predictions.
            labels (Tensor): Ground truth labels.
            rotation (Tensor): Rotation matrix R of shape (D, D).
            binary_reg_weight (float): Weight for the binary regularization term.
            permutation_reg_weight (float): Weight for the permutation regularization term.

        Returns:
            Tensor: The computed total loss.
        """
        # --- Main counterfactual loss (cross entropy / mean square error) ---
        ce = torch.nn.CrossEntropyLoss()
        ce_loss = ce(outputs, labels)

        # --- Binary regularization for all elements in R ---
        # This term is 0 when an element is exactly 0 or 1, and positive otherwise.
        binary_loss = torch.sum((rotation * (1 - rotation)) ** 2)

        # --- Permutation regularization loss on all sub blocks in R ---
        # removed...

        # --- Total Loss ---
        total_loss = ce_loss + binary_reg_weight * binary_loss
        return total_loss

    def batched_random_sampler(data):
        batch_indices = [i for i in range(int(len(data) / batch_size))]
        random.shuffle(batch_indices)
        for b_i in batch_indices:
            for i in range(b_i * batch_size, (b_i + 1) * batch_size):
                yield i

    intervenable.model.train()  # set to train mode for DAS training
    # print("Distributed Intervention Training, trainable parameters: ", intervenable.count_parameters())

    iia = 0.0
    best_iia = 0.0  # Will store the best observed IIA (i.e., DII score)
    total_step = 0

    main_pbar = tqdm(total=epochs * len(counterfactual_train_dataset) // batch_size,
                     desc=f"R Training, Hypothesis {alignment_hypothesis.id}", position=0, leave=True)

    for epoch in range(epochs):
        epoch_iterator = DataLoader(
            counterfactual_train_dataset, batch_size=batch_size, sampler=batched_random_sampler(counterfactual_train_dataset))
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
            base_images = torch.stack([train_mnist[int(idx.item())][0] for idx in base_indices])

            # Retrieve source images from indices.
            # Assume that batch["source_input_ids"] is of shape [batch_size, 2, 1] after unsqueeze.
            source_indices = batch["source_input_ids"].squeeze(2)  # shape: [batch_size, 2]
            source_images = []
            # Loop over the two source positions (e.g., one for C1 and one for C2).
            for pos in range(source_indices.shape[1]):
                imgs = torch.stack([train_mnist[int(idx.item())][0] for idx in source_indices[:, pos]])
                source_images.append(imgs)

            # Call the intervenable model depending on the intervention_id.
            intervention_id = batch["intervention_id"][0]

            counterfactual_outputs = apply_intervention(intervenable, base_images, source_images, intervention_id, batch_size, alignment_hypothesis)

            # Compute batch metrics
            eval_metrics = compute_metrics(counterfactual_outputs[0].detach(), batch["labels"].squeeze())
            R_params = next(iter(intervenable.interventions.values()))[0].rotate_layer.weight  # R shape: [h_dim, h_dim] (expected 20x20)
            loss = compute_loss(counterfactual_outputs[0], batch["labels"].squeeze().to(torch.long), R_params,
                                binary_reg_weight, permutation_reg_weight)

            # Update the progress bar with batch metrics
            main_pbar.set_postfix({"batch loss": f"{loss.item():.4f}", "batch acc": f"{eval_metrics['accuracy']:.4f}"})
            main_pbar.set_postfix(
                {"batch loss": f"{loss.item():.2f}", "batch acc": f"{eval_metrics['accuracy']:.2f}", "val IIA": f"{iia:.4f}",
                 "DII score": f"{best_iia:.4f}"})
            main_pbar.update(1)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if total_step % gradient_accumulation_steps == 0:
                optimizer.step()
                intervenable.set_zero_grad()
            total_step += 1

        scheduler.step()  # decay learning rate

        """Validation after each epoch"""
        intervenable.model.eval()  # set model to evaluation mode
        total_correct, total_samples = 0, 0

        eval_dataloader = DataLoader(counterfactual_val_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in eval_dataloader:
                # Retrieve base images.
                base_indices = batch["input_ids"].squeeze(1)
                base_images = torch.stack([val_mnist[int(idx.item())][0] for idx in base_indices])

                # Retrieve source images.
                source_indices = batch["source_input_ids"].squeeze(2)
                source_images = []
                for pos in range(source_indices.shape[1]):
                    imgs = torch.stack([val_mnist[int(idx.item())][0] for idx in source_indices[:, pos]])
                    source_images.append(imgs)

                # Apply the intervention.
                intervention_id = batch["intervention_id"][0]
                counterfactual_outputs = apply_intervention(
                    intervenable, base_images, source_images, intervention_id, batch_size, alignment_hypothesis)

                # Compute metrics for this batch.
                metrics = compute_metrics(counterfactual_outputs[0], batch["labels"].squeeze())
                total_correct += metrics["correct"]
                total_samples += metrics["total"]

        iia = total_correct / total_samples if total_samples > 0 else 0.0
        is_new_best = iia > best_iia
        # Update the best IIA (which we define as DII score) if current epoch accuracy is higher.
        if is_new_best:
            best_iia = iia
            # Retrieve the rotation matrix stored in rotate_layer.weight of all interventions
            R = next(iter(intervenable.interventions.values()))[0].rotate_layer.weight.detach().cpu()
            if save_dir is not None:
                R_path = os.path.join(save_dir, f"R{alignment_hypothesis.id}.bin")
            else:
                R_path = os.path.splitext(state_dict_path)[0] + f"_R{alignment_hypothesis.id}.bin"
            torch.save(R, R_path)

        # Update progress bar with validation results
        main_pbar.set_postfix(
            {"batch loss": f"{loss.item():.2f}", "batch acc": f"{eval_metrics['accuracy']:.2f}", "val IIA": f"{iia:.4f}",
             "DII score": f"{best_iia:.4f}"})

        # Switch back to training mode
        intervenable.model.train()

    main_pbar.update(1)
    main_pbar.close()
    print(f"DII score {best_iia:.4f} ({state_dict_path}) with hypothesis {alignment_hypothesis.id}: {alignment_hypothesis.description})")
    return best_iia, R_path


def iterate_hypothesis(target_model, state_dict_path, hypotheses_to_test=range(1, 11), base_dir="trained_models"):
    """
    Create a directory for the given model name, copy the pretrained model there,
    and iterate over different alignment hypotheses. For each hypothesis it calls distributed_alignment_search(),
    logs the best DII score and saves the best rotation matrix.

    A CSV file with one row per hypothesis is saved in the model directory.
    """
    # Create output directory describing the used method and architecture
    dir_name = f"{args.model}_pairs" if args.joint else f"{args.model}_single"
    if "csup" in state_dict_path:
        dir_name += "_csup"

    model_dir = os.path.join(base_dir, dir_name)
    os.makedirs(model_dir, exist_ok=True)

    # Copy pretrained model to model_dir as model.pth (if available)
    if os.path.isfile(state_dict_path):
        shutil.copy(state_dict_path, os.path.join(model_dir, "model.pth"))
    else:
        print("State dict path does not exist as a file. Skipping copy of model.pth.")

    results = []

    # Iterate over hypotheses: range(1, 11) -> i from 1 to 10
    for i in hypotheses_to_test:
        hypothesis = AlignmentHypothesis(i)
        print(f"Evaluating hypothesis {i}: {hypothesis.description}")

        # Call DAS with the current hypothesis.
        # The modified DAS accepts an optional alignment_hypothesis as well as output parameters.
        best_iia, R_path = distributed_alignment_search(
            target_model,
            state_dict_path,
            alignment_hypothesis=hypothesis,
            save_dir=model_dir
        )

        results.append({
            "hypothesis_id": hypothesis.id,
            "C1_dims": hypothesis.C1_dims,
            "C2_dims": hypothesis.C2_dims,
            "DII_score": best_iia,
            "R_path": R_path
        })

    # Write the results to a CSV file inside the model directory.
    csv_path = os.path.join(model_dir, "alignment_results.csv")
    with open(csv_path, mode="w", newline="") as csv_file:
        fieldnames = ["hypothesis_id", "DII_score", "C1_dims", "C2_dims", "R_path"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Iterate over all hypothesis complete. Results saved to {csv_path}")


def eval_DAS_alignment(target_model, state_dict_path, data_split: str, saved_R_path=None):
    """
    This method loads a target model with a trained rotation matrix and evaluates its alignment. How good is the target_model
    with the learned rotation matrix to predict counterfactual data (of different splits) produced by the causal abstraction model.
    Args:
        target_model: Model architecture to use
        state_dict_path: Trained target_model parameters
        data_split: "train", "val", "test" data split to evaluate on
        saved_R_path: Path to the saved rotation matrix '.bin' file (torch saved tensor).
                      If None, try to find corresponding R given the state_dict_path

    Returns: None - prints the classification statistics
    """
    # Load the target model
    target_model.load_state_dict(torch.load(state_dict_path))
    target_model.eval()
    target_model.device = "cpu"
    target_model.to(target_model.device)
    print(target_model)

    # Build the intervenable
    intervenable = init_intervenable(target_model)

    # Load the rotation matrix (ensure CPU, detach, convert to NumPy)
    if saved_R_path is None:  # If None, try to find corresponding R given the state_dict_path
        saved_R_path = os.path.splitext(state_dict_path)[0] + "_R.bin"
    R = torch.load(saved_R_path, weights_only=True).cpu().detach().numpy()

    # Copy the given rotation matrix into intervention[0].rotate_layer
    with torch.no_grad():
        for key, intervention in intervenable.interventions.items():
            state = intervention[0].rotate_layer.state_dict()
            # Convert R to a tensor with the proper type and device.
            R_tensor = torch.tensor(R, dtype=intervention[0].rotate_layer.weight.dtype, device=target_model.device)
            # Now force the rotate layer to load the given state.
            state['parametrizations.weight.0.base'] = R_tensor.clone()
            intervention[0].rotate_layer.load_state_dict(state)

    # Freeze the entire model to ensure no parameters are updated during evaluation
    for param in intervenable.model.parameters():
        param.requires_grad = False

    # Load the dataset to retrieve image tensors based on the split.
    if data_split == 'train':
        mnist = train_mnist
        counterfactual = counterfactual_train
    elif data_split == 'val':
        mnist = val_mnist
        counterfactual = counterfactual_val
    elif data_split == 'test':
        mnist = test_mnist
        counterfactual = counterfactual_test
    else:
        raise ValueError(f"Invalid split: {data_split}")

    def extract_batch_size(path: str) -> int:
        for part in path.replace('.', '_').split('_'):
            if part.startswith('bs') and part[2:].isdigit():
                return int(part[2:])
        raise ValueError("Batch size not extracted from counterfactual data path")

    def extract_hypothesis(path: str) -> int:
        parts = re.split(r"[._/-]+", path)
        for part in parts:
            if part.startswith('R') and part[1:].isdigit():
                return int(part[1:])

        raise ValueError("Hypothesis ID not extracted from saved R path")

    bs = extract_batch_size(counterfactual)  # Assure we use the correct bs that was used for counterfactual data generation
    alignment_hypothesis = AlignmentHypothesis(extract_hypothesis(saved_R_path))  # Get AlignmentHypothesis used for this R

    # Load the counterfactual data
    print(f"loading counterfactual {data_split} data")
    counterfactual_dataset = torch.load(counterfactual, weights_only=True)

    eval_labels = []
    eval_preds = []
    eval_dataloader = DataLoader(counterfactual_dataset, batch_size=bs, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating"):
            # Retrieve base images.
            base_indices = batch["input_ids"].squeeze(1)
            base_images = torch.stack([mnist[int(idx.item())][0] for idx in base_indices])

            # Retrieve source images.
            source_indices = batch["source_input_ids"].squeeze(2)
            source_images = []
            for pos in range(source_indices.shape[1]):
                imgs = torch.stack([mnist[int(idx.item())][0] for idx in source_indices[:, pos]])
                source_images.append(imgs)

            # Apply the intervention.
            intervention_id = batch["intervention_id"][0]
            counterfactual_outputs = apply_intervention(
                intervenable, base_images, source_images, intervention_id, bs, alignment_hypothesis)

            # Compute metrics for this batch.
            eval_labels += [batch["labels"]]
            eval_preds += [torch.argmax(counterfactual_outputs[0], dim=1)]
    print(classification_report(torch.cat(eval_labels).cpu(), torch.cat(eval_preds).cpu(), digits=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Alignment Search", allow_abbrev=False)
    parser.add_argument(
        "--model",
        type=str,
        default="mnistdpl",
        help="Target model to intervene on",
        choices=["mnistdpl", "mnistnn"],
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
        action="store_true",
        help="Set this flag accordingly to which model you want to load: Use joint architecture (PairsEncoder)."
             "If not set, SingleEncoder of digit images is used.",
    )
    parser.add_argument(
        "--iterate_hyp",
        action="store_true",
        help="If set, run multiple iterations of DAS to evaluate a range of alignment hypothesis",
    )
    # Evaluation-specific arguments
    parser.add_argument(
        "--only_eval",
        action="store_true",
        help="If set, run evaluation of a learned rotation matrix R using eval_DAS_alignment instead of DAS training",
    )
    parser.add_argument(
        "--saved_R",
        type=str,
        default=None,
        help="Path to the saved rotation matrix (.bin file) for eval_DAS_alignment. If None, try to find corresponding R given the state_dict_path",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Data split for eval_DAS_alignment (train/val/test)",
    )
    args = parser.parse_args()

    dataset = get_dataset(args)
    encoder, decoder = dataset.get_backbone()  # get joint or disjoint architecture
    n_images, c_split = dataset.get_split()  # Based on args.joint True or False: n_images= 1 or 2 c_split= (10, 10) or (10,)
    model = get_model(args, encoder, decoder, n_images, c_split)

    # Load Mnist images
    train_mnist, val_mnist, test_mnist = load_2MNIST(args=SimpleNamespace(task="addition"))

    # Your path to the counterfactual data (mapping of image indices to counterfactual predictions) to evaluate DAS on
    counterfactual_train = "data/mnist_add_counterfactual_train_data_bs100.pt"
    counterfactual_val = "data/mnist_add_counterfactual_val_data_bs100.pt"
    counterfactual_test = "data/mnist_add_counterfactual_test_data_bs100.pt"

    pretrained_path = args.pretrained
    # Check if model name is in pretrained path
    if args.model not in pretrained_path:
        raise ValueError(f"Model name '{args.model}' not found in pretrained path '{pretrained_path}'")
    # Check if 'PairsEncoder' or 'SingleEncoder' is correctly specified
    expected_encoder = "PairsEncoder" if args.joint else "SingleEncoder"
    if expected_encoder not in pretrained_path:
        raise ValueError(f"Expected '{expected_encoder}' in pretrained path '{pretrained_path}', but not found.")

    if args.only_eval:
        """
        Evaluate the alignment using the learned rotation matrix.
        Example call:
        python DAS.py --only_eval --pretrained trained_models/mnistdpl_MNISTPairsEncoder.pth --joint --data_split val
        """
        eval_DAS_alignment(
            target_model=model,
            state_dict_path=pretrained_path,
            data_split=args.data_split,
            saved_R_path=args.saved_R,
        )
    elif args.iterate_hyp:
        """
        Use DAS multiple times to iterate over a range of alignment hypothesis
        Example call:
        python DAS.py --iterate_hyp --pretrained trained_models/mnistdpl_MNISTPairsEncoder.pth --joint
        """
        print("Loading counterfactual data")
        counterfactual_train_dataset = torch.load(counterfactual_train, weights_only=True)
        counterfactual_val_dataset = torch.load(counterfactual_val, weights_only=True)
        iterate_hypothesis(
            target_model=model,
            state_dict_path=pretrained_path,
        )
    else:
        """
        Train the rotation matrix using DAS.
        Example call:
        python DAS.py --pretrained trained_models/mnistdpl_MNISTPairsEncoder.pth --joint
        """
        # Load the needed counterfactual data
        print("Loading counterfactual data")
        counterfactual_train_dataset = torch.load(counterfactual_train, weights_only=True)
        counterfactual_val_dataset = torch.load(counterfactual_val, weights_only=True)
        distributed_alignment_search(
            target_model=model,
            state_dict_path=pretrained_path,
        )
