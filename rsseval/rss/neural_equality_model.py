import torch
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random
import copy
import itertools
import numpy as np
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


def train_mlp(config, dataset_path: str):
    mlp_config = MLPConfig(**config)
    mlp_config, tokenizer, mlp = create_mlp_classifier(mlp_config)
    mlp.train()

    print("loading train data")
    dataset = torch.load(dataset_path, weights_only=True)
    X = torch.stack([example['input_ids'] for example in dataset])
    y = torch.stack([example['labels'] for example in dataset])

    train_ds = Dataset.from_dict(
        {
            "labels": [
                torch.FloatTensor([0, 1]) if i == 1 else torch.FloatTensor([1, 0])
                for i in y
            ],
            "inputs_embeds": X,
        }
    )

    training_args = TrainingArguments(
        output_dir="data/equality_model_trainer",
        evaluation_strategy="epoch",
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        report_to="none",
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
    print("training MLP")
    _ = trainer.train()
    return trainer


def test_mlp(trainer, dataset_path: str):
    print("loading test data")
    test_dataset = torch.load(dataset_path, weights_only=True)
    X_test = torch.stack([example['input_ids'] for example in test_dataset])
    y_test = torch.stack([example['labels'] for example in test_dataset])

    # Create the test dataset for evaluation
    test_ds = Dataset.from_dict(
        {
            "labels": [
                torch.FloatTensor([0, 1]) if label.item() == 1 else torch.FloatTensor([1, 0])
                for label in y_test
            ],
            "inputs_embeds": X_test,
        }
    )

    print("evaluating model")
    test_preds = trainer.predict(test_ds)

    y_pred = test_preds.predictions.argmax(axis=1)  # Predicted class labels
    y_true = [example['labels'].item() for example in test_dataset]  # True class labels

    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    object_dim = 2

    args = dict(
        num_classes=2,
        n_layer=3,
        h_dim=object_dim * 4,
        activation_function="relu",
        learning_rate=0.01,
        num_train_epochs=3,
        batch_size=1024,
        pdrop=0.0,

    )
    trainer = train_mlp(args, dataset_path="data/equality_task_data_dim2.pt")

    test_mlp(trainer, dataset_path="data/equality_task_data_dim2.pt")
