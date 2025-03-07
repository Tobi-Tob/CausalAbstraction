import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datasets.utils.base_dataset import BaseDataset
from utils.metrics import evaluate_metrics, evaluate_mix


def test(model, dataset: BaseDataset, args):
    """Evaluation of model on test set, looks at label and concept accuracy.
    This function saves a graphic that contains the label and concept accuracy
    and their confusion matrices.
    The confusion matrix is a table that compares a model's predictions with the actual class labels.
    Each row represents the true class, while each column represents the predicted class,
    making it easy to see which classes are being correctly identified and where misclassifications occur.
    Args:
        model (MnistDPL): network architecture to test
        dataset (BaseDataset): dataset
        args: parsed args

    Returns:
        None
    """
    model_path = args.checkin
    # Check if model name is in model path
    if model_path is not None and args.model not in model_path:
        raise ValueError(f"Model name '{args.model}' not found in model path '{model_path}'")
    # Check if 'PairsEncoder' or 'SingleEncoder' is correctly specified
    expected_encoder = "PairsEncoder" if args.joint else "SingleEncoder"
    if model_path is not None and expected_encoder not in model_path:
        raise ValueError(f"Expected '{expected_encoder}' in model path '{model_path}', but not found.")

    # Load the model or initialize randomly
    if model_path is None:
        print("Random Initialization of Model")
        model_path = "random_baseline"
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    model.device = "cpu"
    model.to(model.device)

    _, _, test_loader = dataset.get_data_loaders()

    print("--- Start of Test Evaluation ---")
    # Evaluate metrics: assuming these functions return
    # true labels, true concepts, predicted labels, and predicted concepts.
    y_true, c_true, y_pred, c_pred = evaluate_metrics(model, test_loader, args, last=True)
    yac, yf1 = evaluate_mix(y_true, y_pred)
    cac, cf1 = evaluate_mix(c_true, c_pred)

    print(f'Concepts:\n    ACC: {cac}, F1: {cf1}')
    print(f'Labels:\n      ACC: {yac}, F1: {yf1}')

    # Create confusion matrices using scikit-learn
    cm_labels = confusion_matrix(y_true, y_pred)
    cm_concepts = confusion_matrix(c_true, c_pred)

    # Get the number of classes based on the confusion matrix shape
    num_label_classes = cm_labels.shape[0]
    num_concept_classes = cm_concepts.shape[0]

    # Create a figure with two subplots: one for labels, one for concepts
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the label confusion matrix
    sns.heatmap(cm_labels, annot=True, fmt="d", cmap="Blues",
                xticklabels=[str(i) for i in range(num_label_classes)],
                yticklabels=[str(i) for i in range(num_label_classes)],
                ax=axs[0])
    axs[0].set_title(f"Label Confusion Matrix\nACC: {yac * 100:.2f}%, F1: {yf1 * 100:.2f}%")
    axs[0].set_xlabel("Predicted Label")
    axs[0].set_ylabel("True Label")

    # Plot the concept confusion matrix
    sns.heatmap(cm_concepts, annot=True, fmt="d", cmap="Greens",
                xticklabels=[str(i) for i in range(num_concept_classes)],
                yticklabels=[str(i) for i in range(num_concept_classes)],
                ax=axs[1])
    axs[1].set_title(f"Concept Confusion Matrix\nACC: {cac * 100:.2f}%, F1: {cf1 * 100:.2f}%")
    axs[1].set_xlabel("Predicted Concept")
    axs[1].set_ylabel("True Concept")
    plt.tight_layout()

    if model_path != "random_baseline":
        root, ext = os.path.splitext(model_path)
        output_filename = f"{root}_eval.png"
    else:
        output_filename = f"{args.model}_evaluation.png"
    plt.savefig(output_filename)
    print(f"Saved evaluation results to {output_filename}")

