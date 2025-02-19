import os
import numpy as np
import torch
from numpy import linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt


def eigenvalues_to_angles(eigenvalues):
    angles = []
    for eig in eigenvalues:
        angle = np.arctan2(np.imag(eig), np.real(eig))
        angles.append(angle)
    return angles


def to_degree_angles(angles):
    # Convert angles from radians to absolute degrees.
    degree_angles = [abs(np.degrees(angle)) for angle in angles]
    return degree_angles


def check_orthogonality(R, tol=1e-6):
    """
    Checks if a given rotation matrix R is orthogonal and orthonormal.

    Parameters:
        R (torch.Tensor or np.ndarray): The rotation matrix.
        tol (float): Tolerance for numerical precision errors.

    Returns:
        (bool, bool): (is_orthogonal, is_orthonormal)
    """
    if isinstance(R, torch.Tensor):
        R = R.cpu().detach().numpy()  # Convert to NumPy for easier manipulation

    # Compute R^T * R
    RtR = np.dot(R.T, R)
    identity = np.eye(R.shape[0])

    # Check orthogonality: R^T * R should be close to the identity matrix
    is_orthogonal = np.allclose(RtR, identity, atol=tol)

    # Check orthonormality: Each column should have a norm of 1
    column_norms = np.linalg.norm(R, axis=0)
    is_orthonormal = is_orthogonal and np.allclose(column_norms, 1.0, atol=tol)

    return is_orthogonal, is_orthonormal


def visualize_rotation_degrees(saved_R_path: str):
    # Load the rotation matrix (ensure it's on CPU and detached)
    R = torch.load(saved_R_path, weights_only=True).cpu().detach().numpy()

    # Compute eigenvalues of the rotation matrix
    eigenvalues, _ = LA.eig(R)

    # Convert eigenvalues to rotation angles (in degrees)
    angles = eigenvalues_to_angles(eigenvalues)
    degree_angles = to_degree_angles(angles)

    # Set up matplotlib style
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams.update({'mathtext.default': 'regular'})

    with plt.rc_context({
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'figure.facecolor': 'white'
    }):
        fig, ax = plt.subplots(figsize=(5, 3.8))

        # Create the histogram plot for the rotation degrees
        sns.histplot(degree_angles, bins=30, ax=ax, color="skyblue")

        ax.set_xlabel("Basis Vector Rotation Degree(s)", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.legend(labels=["Rotation Matrix"], loc="upper right")
        ax.grid(color='grey', linestyle='-.', linewidth=1, alpha=0.5)

        plt.tight_layout()
        img_path = os.path.splitext(saved_R_path)[0] + "_degrees.png"
        plt.savefig(img_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved rotation degrees histogram to: {img_path}")


def visualize_rotation_matrix(saved_R_path: str):
    """
    Loads a rotation matrix from a file, produces a heatmap of it,
    and saves the heatmap image.

    Args:
        saved_R_path (str): Path to the saved rotation matrix '.bin' file (torch saved tensor).
    """
    # Load the rotation matrix (ensure it's on CPU and detached)
    R = torch.load(saved_R_path, weights_only=True).cpu().detach().numpy()

    # Check matrix properties
    is_orthogonal, is_orthonormal = check_orthogonality(R)

    # Create a heatmap of the rotation matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(R, cmap="viridis", cbar=True)
    plt.title("Heatmap of the Rotation Matrix")
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    # Add matrix properties as text below the heatmap
    text_props = f"Orthogonal: {is_orthogonal}\nOrthonormal: {is_orthonormal}"
    plt.figtext(0.5, -0.1, text_props, wrap=True, horizontalalignment='center', fontsize=12)

    # Adjust layout and save the heatmap to file
    plt.tight_layout()
    img_path = os.path.splitext(saved_R_path)[0] + "_heatmap.png"
    plt.savefig(img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved rotation matrix heatmap to: {img_path}")


def visualize_concept_contribution(saved_R_path: str):
    """
    Loads a square rotation matrix R, computes each original dimension's
    contribution to concept C1 (sum of absolute values in rows 0..half-1) and C2 (rows half..end),
    then produces a stacked bar chart (red=C1, blue=C2).

    Visualization idea:
    We have a concept vector (representing the neural activations before rotation), and a rotation matrix R that,
    once applied, yields a new vector whose first half of the entries correspond to concept C1 (red) and
    last half of the entries correspond to concept C2 (blue).
    The key idea is to show before the rotation which dimensions of the original concept vector “belong” more to C1 vs. C2.
    It visually explains which parts (dimensions) of the original concept representation are more responsible for the final
    “red” concept (C1) vs. the final “blue” concept (C2).

    Args:
        saved_R_path (str): Path to the saved rotation matrix '.bin' file (torch saved tensor).
    """
    # Load the rotation matrix (ensure CPU, detach, convert to NumPy)
    R = torch.load(saved_R_path, weights_only=True).cpu().detach().numpy()

    # Ensure it's a square matrix
    num_dims = R.shape[0]
    assert R.shape[0] == R.shape[1], "Rotation matrix R must be square (NxN)."
    half = num_dims // 2

    # Prepare arrays to store normalized contributions for each dimension
    c1_contribution = np.zeros(num_dims)
    c2_contribution = np.zeros(num_dims)

    # Compute the contribution for each column (dimension)
    for i in range(num_dims):
        # Column i of R
        col = R[:, i]

        # Sum absolute values for top/bottom halves
        c1_contribution_of_dim_i = np.sum(np.abs(col[:half]))  # C1 (first half rows)
        c2_contribution_of_dim_i = np.sum(np.abs(col[half:]))  # C2 (last half rows)

        # Normalize
        denom = c1_contribution_of_dim_i + c2_contribution_of_dim_i + 1e-12
        c1_contribution[i] = c1_contribution_of_dim_i / denom
        c2_contribution[i] = c2_contribution_of_dim_i / denom

    # Plot a stacked bar chart
    indices = np.arange(num_dims)
    plt.figure(figsize=(10, 6))

    # Bottom part of the bar = c1_contribution (C1)
    plt.bar(indices, c1_contribution, color='red', label='C1')

    # Top part of the bar = c2_contribution (C2), stacked on top of c1_contribution
    plt.bar(indices, c2_contribution, bottom=c1_contribution, color='blue', label='C2')

    # Cosmetics
    plt.xticks(indices, [f"Dim {i}" for i in range(num_dims)], rotation=90)
    plt.ylim([0, 1])
    plt.xlabel("Original Dimension Index", fontsize=12)
    plt.ylabel("Normalized Concept Contribution", fontsize=12)
    plt.title("C1 vs C2 Concept Contribution for each Original Dimension", fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Construct the output path
    img_path = os.path.splitext(saved_R_path)[0] + "_contribution.png"
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved concept contribution bar chart to: {img_path}")


def build_test_R(safe_location):
    R_test = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.1, 0.5],
        [0.0, 0.1, 0.9, 0.0],
        [0.0, 0.5, 0.0, 0.5]
    ])
    R_test_tensor = torch.tensor(R_test, dtype=torch.float32)
    torch.save(R_test_tensor, safe_location)


if __name__ == "__main__":
    # build_test_R("trained_models/test_R.bin")
    load_R = "trained_models/mnistdpl_MNISTSingleEncoder_0.0_None_R.bin"
    visualize_rotation_degrees(load_R)
    visualize_rotation_matrix(load_R)
    visualize_concept_contribution(load_R)
