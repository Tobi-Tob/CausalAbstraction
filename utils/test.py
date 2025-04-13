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
    # y_true = true labels of data size
    # c_true = true concepts of size 2x datasize as concatenation of C1,C2
    # y_pred = predicted labels of data size
    # c_pred = predicted concepts of size 2x datasize as concatenation of C1,C2
    y_true, c_true, y_pred, c_pred = evaluate_metrics(model, test_loader, args, last=True)
    yac, yf1 = evaluate_mix(y_true, y_pred)
    cac, cf1 = evaluate_mix(c_true, c_pred)

    print(f'Concepts:\n    ACC: {cac}, F1: {cf1}')
    print(f'Labels:\n      ACC: {yac}, F1: {yf1}')

    # Create confusion matrices using scikit-learn
    cm_labels = confusion_matrix(y_true, y_pred)
    cm_concepts = confusion_matrix(c_true, c_pred)

    # Compute 100x100 confusion matrix for digit pairs
    # Here we assume that c_true and c_pred are 1D arrays of length 2*N,
    # where the first N entries correspond to the first digit (g1 or c1)
    # and the second N entries correspond to the second digit (g2 or c2).
    N = len(c_true) // 2
    # Encode each digit pair into a unique label: (digit1, digit2) -> digit1*10 + digit2
    true_pairs = (c_true[:N].astype(int) * 10) + c_true[N:].astype(int)
    print(true_pairs)
    pred_pairs = (c_pred[:N].astype(int) * 10) + c_pred[N:].astype(int)
    print(pred_pairs)
    # Create confusion matrix for pairs with all 100 possible labels (0 to 99)
    cm_pairs = confusion_matrix(true_pairs, pred_pairs, labels=list(range(100)))

    # Get the number of classes based on the confusion matrix shape
    num_label_classes = cm_labels.shape[0]
    num_concept_classes = cm_concepts.shape[0]

    # Create a figure with three subplots: labels, concepts, and digit-pair confusion matrices
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

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

    # Plot the 100x100 digit-pair confusion matrix
    sns.heatmap(cm_pairs, annot=False, cmap="Purples", ax=axs[2])
    axs[2].set_xticks(range(0, 100, 5))
    axs[2].set_xticklabels([f"{i // 10},{i % 10}" for i in range(0, 100, 5)], fontsize=8)
    axs[2].set_yticks(range(0, 100, 5))
    axs[2].set_yticklabels([f"{i // 10},{i % 10}" for i in range(0, 100, 5)], fontsize=8)
    axs[2].set_title("Digit-Pair Confusion Matrix")
    axs[2].set_xlabel("Predicted Pair")
    axs[2].set_ylabel("True Pair")

    plt.tight_layout()

    if model_path != "random_baseline":
        root, ext = os.path.splitext(model_path)
        output_filename = f"{root}_eval.png"
    else:
        output_filename = f"{args.model}_evaluation.png"
    plt.savefig(output_filename)
    print(f"Saved evaluation results to {output_filename}")

