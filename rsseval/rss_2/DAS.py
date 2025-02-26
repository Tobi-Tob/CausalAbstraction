import argparse
import os
import random
from types import SimpleNamespace
import torch
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
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
    h_dim = intervenable.model.h_dim
    subspace_idx = int(h_dim / 2)
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
            subspaces=[  # Define the subspace of neurons to target. We align the first half of h_dim to C1 and the second half to C2
                [[i for i in range(0, subspace_idx)]] * batch_size,
                [[i for i in range(subspace_idx, h_dim)]] * batch_size,
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
                [[i for i in range(0, subspace_idx)]] * batch_size,
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
                [[i for i in range(subspace_idx, h_dim)]] * batch_size,
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


def distributed_alignment_search(target_model, state_dict_path, counterfactual_data_path: str, args):
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
        counterfactual_data_path (str): path to the counterfactual data (mapping of image indices to counterfactual predictions)
        args: Containing further training parameters

    Call from terminal:
    python main.py --DAS --model mnistdpl --dataset addmnist --task addition --backbone conceptizer --checkin test_model_addmnist_mnistdpl.pth --batch_size 100
    """

    # ============== Training parameters and config ==============
    epochs = args.epochs
    batch_size = args.bs
    lr = 0.01
    weight_decay = 0.001
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

    # Load the counterfactual data
    print("loading counterfactual data")
    counterfactual_train_dataset = torch.load(counterfactual_data_path, weights_only=True)
    counterfactual_val_dataset = torch.load(counterfactual_val, weights_only=True)

    # Load the dataset to retrieve image tensors
    train_mnist, val_mnist, _ = load_2MNIST(args=SimpleNamespace(task="addition"))

    # Optimizer: we only optimize the rotation parameters from DAS, the rest of the model is frozen.
    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
        break
    optimizer = torch.optim.Adam(optimizer_params, lr=lr, weight_decay=weight_decay)

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
                counterfactual_train_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(counterfactual_train_dataset),
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

        eval_dataloader = DataLoader(counterfactual_val_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc=f"Validating Epoch {epoch}"):
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
                    intervenable, base_images, source_images, intervention_id, batch_size)

                # Compute metrics for this batch.
                metrics = compute_metrics(counterfactual_outputs[0], batch["labels"].squeeze())
                total_correct += metrics["correct"]
                total_samples += metrics["total"]

        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch} Evaluation IIA: {epoch_accuracy:.4f}")

        # Update the best IIA (DII score) if current epoch accuracy is higher.
        if epoch_accuracy > best_iia:
            best_iia = epoch_accuracy
            # Retrieve the rotation matrix from the first intervention (adjust if needed)
            for key, intervention in intervenable.interventions.items():
                # intervention[0] contains the rotation module; get its weight parameter.
                R = intervention[0].rotate_layer.weight.detach().cpu()
                break  # We only need one rotation matrix
            R_path = os.path.splitext(state_dict_path)[0] + "_R.bin"
            torch.save(R, R_path)
            print("New best IIA achieved. Rotation matrix saved.")

        # Switch back to training mode.
        intervenable.model.train()

    print(f"DII score of {state_dict_path} (highest IIA observed): {best_iia:.4f}")


def eval_DAS_alignment(target_model, state_dict_path, bs: int, data_split: str, saved_R_path=None):
    """
    This method loads a target model with a trained rotation matrix and evaluates its alignment. How good is the target_model
    with the learned rotation matrix to predict counterfactual data (of different splits) produced by the causal abstraction model.
    Args:
        target_model: Model architecture to use
        state_dict_path: Trained target_model parameters
        bs: Batch size to use when loading the counterfactual data. Important to have the same bs used for counterfactual data generation!
        data_split: "train", "val", "test" - Call with the same split used for counterfactual data generation to retrieve the correct images!
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
        dataset, _, _ = load_2MNIST(args=SimpleNamespace(task="addition"))
        counterfactual = counterfactual_train
    elif data_split == 'val':
        _, dataset, _ = load_2MNIST(args=SimpleNamespace(task="addition"))
        counterfactual = counterfactual_val
    elif data_split == 'test':
        _, _, dataset = load_2MNIST(args=SimpleNamespace(task="addition"))
        counterfactual = counterfactual_test
    else:
        raise ValueError(f"Invalid split: {data_split}")

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
            base_images = torch.stack([dataset[int(idx.item())][0] for idx in base_indices])

            # Retrieve source images.
            source_indices = batch["source_input_ids"].squeeze(2)
            source_images = []
            for pos in range(source_indices.shape[1]):
                imgs = torch.stack([dataset[int(idx.item())][0] for idx in source_indices[:, pos]])
                source_images.append(imgs)

            # Apply the intervention.
            intervention_id = batch["intervention_id"][0]
            counterfactual_outputs = apply_intervention(
                intervenable, base_images, source_images, intervention_id, bs)

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
        "--bs",
        type=int,
        default=100,
        help="Batch size of the counterfactual data sampling. Its important to have the same bs used for counterfactual data generation!",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="How many epochs to train the rotation matrix",
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
        python DAS.py --only_eval --pretrained trained_models/mnistdpl_MNISTPairsEncoder_0.0_None.pth --joint --data_split val
        """
        eval_DAS_alignment(
            target_model=model,
            state_dict_path=pretrained_path,
            bs=args.bs,
            data_split=args.data_split,
            saved_R_path=args.saved_R,
        )
    else:
        """
        Train the rotation matrix using DAS.
        Example call:
        python DAS.py --pretrained trained_models/mnistdpl_MNISTPairsEncoder_0.0_None.pth --joint
        """
        distributed_alignment_search(
            target_model=model,
            state_dict_path=pretrained_path,
            counterfactual_data_path=counterfactual_train,
            args=args,
        )
