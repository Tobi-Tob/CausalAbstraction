import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm, trange
from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup
from pyvene.models.mlp.modelings_mlp import MLPConfig
from pyvene import create_mlp_classifier
from pyvene import (
    IntervenableModel,
    VanillaIntervention,
    RotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def train_mlp(config, dataset_path: str, save_model_dir: str):
    """
    Train the MLP model on the hierarchical equality task data produced by a causal model.

    Args:
        config (dict): Configuration for the MLP model.
        dataset_path (str): Path to the dataset file.
        save_model_dir (str): Directory to save the trained model.
    """
    # Configure and create the MLP model
    mlp_config = MLPConfig(**config)
    mlp_config, tokenizer, mlp = create_mlp_classifier(mlp_config)
    mlp.train()

    print("Loading training data...")
    dataset = torch.load(dataset_path, weights_only=True)
    X = torch.stack([example['input_ids'] for example in dataset])
    y = torch.stack([example['labels'] for example in dataset])

    train_ds = Dataset.from_dict({
        "labels": [torch.FloatTensor([0, 1]) if i == 1 else torch.FloatTensor([1, 0]) for i in y],
        "inputs_embeds": X,
    })

    training_args = TrainingArguments(
        output_dir=os.path.join(save_model_dir, "equality_model_trainer"),
        eval_strategy="epoch",
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        report_to="none",
        save_strategy="epoch",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=mlp,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=train_ds,
        compute_metrics=lambda x: {
            "accuracy": classification_report(
                x[0].argmax(1), x[1].argmax(1), output_dict=True
            )["accuracy"]
        },
    )

    print("Training MLP model...")
    trainer.train()

    # Save the model
    model_path = os.path.join(save_model_dir, f"mlp_dim{config['h_dim']/4}.pt")
    print(f"Saved model to {model_path}")
    torch.save(mlp, model_path)


def test_mlp(model_path: str, testdata_path: str):
    """
    Test a trained MLP model on a given test dataset.

    Args:
        model_path (str): Path to the saved model file.
        testdata_path (str): Path to the test dataset file.
    """
    print(f"Loading model from {model_path}...")
    # Load the trained model
    model = torch.load(model_path, weights_only=False)
    model.eval()

    print(f"Loading test data from {testdata_path}...")
    # Load the test dataset
    test_dataset = torch.load(testdata_path, weights_only=True)
    X_test = torch.stack([example['input_ids'] for example in test_dataset])
    y_test = torch.stack([example['labels'] for example in test_dataset])

    # Create the test dataset
    test_ds = Dataset.from_dict({
        "labels": [torch.FloatTensor([0, 1]) if label.item() == 1 else torch.FloatTensor([1, 0]) for label in y_test],
        "inputs_embeds": X_test,
    })

    print("Evaluating the model...")
    with torch.no_grad():
        outputs = model(inputs_embeds=X_test)  # Model output is a tuple
        logits = outputs[0]  # Extract the logits (first element of the tuple)
        y_pred = logits.argmax(dim=1).tolist()  # Predicted labels
        y_true = y_test.tolist()  # True labels

    print("Classification Report:")
    print(classification_report(y_true, y_pred))


def distributed_alignment_search(target_model_path: str, object_dim: int, counterfactuals_path: str):
    """
    The two high-level variables WX and YZ of the causal model will be encoded somewhere
    in a multidimensional linear subspaces of the first layer in the neural model.
    To learn these subspaces, we define an IntervenableConfig that allows us to target the first layer
    in the network after it has been rotated by an orthogonal matrix R.

    Args:
        target_model_path (str): Path to the target model file.
        object_dim (int): Dimension of the objects in the causal model.
        counterfactuals_path (str): Path to the counterfactuals dataset file.
    """
    print("loading target model")
    target_model = torch.load(target_model_path, weights_only=False)
    print("loading counterfactual data")
    counterfactual_dataset = torch.load(counterfactuals_path, weights_only=True)

    config = IntervenableConfig(
        model_type=type(target_model),
        representations=[
            RepresentationConfig(
                0,  # layer
                "block_output",  # intervention type
                "pos",  # intervention unit is now align with tokens
                1,  # max number of unit
                subspace_partition=None,  # binary partition with equal sizes
                intervention_link_key=0,
            ),
            RepresentationConfig(
                0,  # layer
                "block_output",  # intervention type
                "pos",  # intervention unit is now align with tokens
                1,  # max number of unit
                subspace_partition=None,  # binary partition with equal sizes,
                intervention_link_key=0,
            ),
        ],
        intervention_types=RotatedSpaceIntervention,
    )

    intervenable = IntervenableModel(config, target_model, use_fast=True)
    intervenable.set_device("cpu")
    intervenable.disable_model_gradients()

    # Training parameters for the rotation matrix
    epochs = 10
    lr = 0.001
    batch_size = 6400
    gradient_accumulation_steps = 1
    total_step = 0

    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
        break
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)

    def compute_metrics(eval_preds, eval_labels):
        total_count = 0
        correct_count = 0
        for eval_pred, eval_label in zip(eval_preds, eval_labels):
            total_count += 1
            correct_count += eval_pred == eval_label
        accuracy = float(correct_count) / float(total_count)
        return {"accuracy": accuracy}

    def compute_loss(outputs, labels):
        CE = torch.nn.CrossEntropyLoss()
        return CE(outputs, labels)

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

    for epoch in train_iterator:
        epoch_iterator = tqdm(
            DataLoader(
                counterfactual_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(counterfactual_dataset),
            ),
            desc=f"Epoch: {epoch}", position=0, leave=True)
        for batch in epoch_iterator:
            batch["input_ids"] = batch["input_ids"].unsqueeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2)
            batch_size = batch["input_ids"].shape[0]
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to("cpu")

            if batch["intervention_id"][0] == 2:
                _, counterfactual_outputs = intervenable(
                    {"inputs_embeds": batch["input_ids"]},
                    [
                        {"inputs_embeds": batch["source_input_ids"][:, 0]},
                        {"inputs_embeds": batch["source_input_ids"][:, 1]},
                    ],
                    {
                        "sources->base": (
                            [[[0]] * batch_size, [[0]] * batch_size],
                            [[[0]] * batch_size, [[0]] * batch_size],
                        )
                    },
                    subspaces=[
                        [[_ for _ in range(0, object_dim * 2)]] * batch_size,
                        [[_ for _ in range(object_dim * 2, object_dim * 4)]]
                        * batch_size,
                    ],
                )
            elif batch["intervention_id"][0] == 0:
                _, counterfactual_outputs = intervenable(
                    {"inputs_embeds": batch["input_ids"]},
                    [{"inputs_embeds": batch["source_input_ids"][:, 0]}, None],
                    {
                        "sources->base": (
                            [[[0]] * batch_size, None],
                            [[[0]] * batch_size, None],
                        )
                    },
                    subspaces=[
                        [[_ for _ in range(0, object_dim * 2)]] * batch_size,
                        None,
                    ],
                )
            elif batch["intervention_id"][0] == 1:
                _, counterfactual_outputs = intervenable(
                    {"inputs_embeds": batch["input_ids"]},
                    [None, {"inputs_embeds": batch["source_input_ids"][:, 0]}],
                    {
                        "sources->base": (
                            [None, [[0]] * batch_size],
                            [None, [[0]] * batch_size],
                        )
                    },
                    subspaces=[
                        None,
                        [[_ for _ in range(object_dim * 2, object_dim * 4)]]
                        * batch_size,
                    ],
                )
            eval_metrics = compute_metrics(
                counterfactual_outputs[0].argmax(1), batch["labels"].squeeze()
            )

            # loss and backprop
            loss = compute_loss(
                counterfactual_outputs[0], batch["labels"].squeeze().to(torch.long)
            )

            epoch_iterator.set_postfix({"loss": loss, "acc": eval_metrics["accuracy"]})

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if total_step % gradient_accumulation_steps == 0:
                optimizer.step()
                intervenable.set_zero_grad()
            total_step += 1


if __name__ == "__main__":
    object_dim = 2

    mlp_args = dict(
        num_classes=2,
        n_layer=3,
        h_dim=object_dim * 4,
        activation_function="relu",
        learning_rate=0.01,
        num_train_epochs=10,
        batch_size=1024,
        pdrop=0.0,

    )
    # train_mlp(mlp_args, dataset_path="data/equality_task_data_dim2.pt", save_model_dir="trained_models")

    # test_mlp(model_path='trained_models/mlp_dim2.pt', testdata_path="data/equality_task_data_dim2.pt")

    distributed_alignment_search(target_model_path='trained_models/mlp_dim2.pt', object_dim=object_dim,
                                 counterfactuals_path="data/equality_task_counterfactual_data_dim2.pt")
