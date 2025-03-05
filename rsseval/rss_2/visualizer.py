import os
import re

import numpy as np
import torch
from numpy import linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt
from DAS import AlignmentHypothesis


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


def extract_hypothesis(path: str) -> int:
    """
    Extracts the hypothesis ID from the file path.
    Assumes the file name contains a segment like "R10" where 10 is the hypothesis ID.
    If no such segment is found, returns None.
    """
    # Split the path using '.', '_', '/' or '-' as delimiters.
    parts = re.split(r"[._/-]+", path)
    for part in parts:
        if part.startswith('R') and part[1:].isdigit():
            return int(part[1:])
    return None


def visualize_rotation_degrees(saved_R_path: str):
    """
    Visualizes the rotation degrees derived from the eigenvalues of a learned rotation matrix R.

    This function loads a saved rotation matrix from disk, computes its eigenvalues,
    and then converts these eigenvalues into rotation angles. A histogram is then generated
    to display the distribution of these rotation angles in degrees.

    Why is this visualization meaningful?
    -----------------------------------------
    1. **Interpreting Rotation Angles:**
       The angles derived from the eigenvalues correspond to the rotation applied in each
       invariant subspace. By visualizing the distribution, one can check whether most rotations
       are centered around specific angles, potentially reflecting the separation into high-level
       concept subspaces.

    2. **Diagnosing Alignment Quality:**
       A concentrated histogram (most angles within a narrow range) suggests a coherent alignment,
       whereas a wide spread could indicate that different basis vectors are rotated by varying degrees,
       which might be a sign of either complex behavior or imperfect alignment.

    3. **Insight into Model Behavior:**
       If certain clusters or outliers are visible, it might hint at invariant directions (e.g., angles
       near 0 or 180 degrees) where the transformation is minimal or simply flipping the sign. Such insights
       can be linked back to whether the model has learned the intended causal abstractions.

    Parameters:
    -----------
    saved_R_path : str
        Path to the saved rotation matrix. The matrix should be stored as a PyTorch tensor.

    Output:
    --------
    - Saves a histogram image of the rotation angles (in degrees) to disk.
    - Prints the eigenvalue magnitudes for a quick numerical check.
    """

    # Load the rotation matrix R, ensuring it is on CPU and detached from any computation graph.
    R = torch.load(saved_R_path, weights_only=True).cpu().detach().numpy()

    # Compute eigenvalues of the rotation matrix.
    eigenvalues, _ = LA.eig(R)
    print("Eigenvalue Magnitudes:", np.abs(eigenvalues))

    # Convert eigenvalues to rotation angles (in radians) and then to degrees.
    angles = eigenvalues_to_angles(eigenvalues)
    degree_angles = to_degree_angles(angles)

    # Set up matplotlib style for a clear, publication-ready plot.
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams.update({'mathtext.default': 'regular'})

    # Use a custom rc_context for styling specific elements of the plot.
    with plt.rc_context({
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'figure.facecolor': 'white'
    }):
        fig, ax = plt.subplots(figsize=(5, 3.8))

        # Create the histogram plot for the rotation degrees.
        # This shows the frequency distribution of the rotation angles corresponding to the eigen-directions.
        sns.histplot(degree_angles, bins=30, ax=ax, color="skyblue")

        # Label axes to convey the meaning of the plotted data.
        ax.set_xlabel("Basis Vector Rotation Degree(s)", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        # ax.legend(labels=["Rotation Matrix"], loc="upper right")
        ax.grid(color='grey', linestyle='-.', linewidth=1, alpha=0.5)

        plt.tight_layout()
        # Save the figure with a descriptive name.
        img_path = os.path.splitext(saved_R_path)[0] + "_degrees.png"
        plt.savefig(img_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved rotation degrees histogram to: {img_path}")


def visualize_rotation_matrix(saved_R_path: str):
    """
    Loads a rotation matrix from a file, produces a heatmap of it,
    and saves the heatmap image. The color scale is fixed to [-1, 1]
    so that 0 appears near white and large negative/positive values
    show strong colors.
    The tick labels on the x and y axes (which represent the row/column indices)
    are colored according to the alignment hypothesis:
      - Red for indices belonging to C1
      - Blue for indices belonging to C2
      - Grey for indices corresponding to None

    Args:
        saved_R_path (str): Path to the saved rotation matrix '.bin' file (torch saved tensor).
    """
    # Load the rotation matrix (ensure it's on CPU and detached)
    # If your torch.save call didn't use `weights_only=True`, remove that argument below
    R = torch.load(saved_R_path, weights_only=True).cpu().detach().numpy()

    # Attempt to extract the hypothesis ID
    hypothesis_id = extract_hypothesis(saved_R_path)
    # If no ID is extracted, assume full concept split: i = h_dim // 2
    if hypothesis_id is None:
        hypothesis_id = R.shape[0] // 2  # Assumes even h_dim
    # Create the AlignmentHypothesis object using the determined ID.
    alignment_hypothesis = AlignmentHypothesis(hypothesis_id, h_dim=R.shape[0])

    # Check matrix properties
    is_orthogonal, is_orthonormal = check_orthogonality(R)

    # Create a heatmap of the rotation matrix using a diverging colormap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        R,
        cmap="coolwarm",
        cbar=True,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=1.0,
        linecolor="black"
    )

    plt.title("Heatmap of the Rotation Matrix")
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    # Color the tick labels on the x-axis.
    for tick in ax.get_xticklabels():
        try:
            idx = int(tick.get_text())
        except ValueError:
            continue  # Skip if the text cannot be converted to an integer.
        if idx in alignment_hypothesis.C1_dims:
            tick.set_color("red")
        elif idx in alignment_hypothesis.C2_dims:
            tick.set_color("blue")
        else:
            tick.set_color("grey")

    # Color the tick labels on the y-axis.
    for tick in ax.get_yticklabels():
        try:
            idx = int(tick.get_text())
        except ValueError:
            continue
        if idx in alignment_hypothesis.C1_dims:
            tick.set_color("red")
        elif idx in alignment_hypothesis.C2_dims:
            tick.set_color("blue")
        else:
            tick.set_color("grey")

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
    Loads a square rotation matrix R, computes each original dimension's contribution to concepts C1, C2, and None,
    then produces a stacked bar chart with red=C1, blue=C2, and grey=None.

    Visualization idea:
    We have a concept vector (representing the neural activations before rotation), and a rotation matrix R that,
    once applied, yields a new vector whose first half of the entries correspond to concept C1 (red) and
    last half of the entries correspond to concept C2 (blue).
    The key idea is to show before the rotation which dimensions of the original concept vector “belong” more to C1 vs. C2.
    It visually explains which parts (dimensions) of the original concept representation are more responsible for the final
    “red” concept (C1) vs. the final “blue” concept (C2).

    It matches the mental model that R is simply a linear transformation redistributing the old dimensions into new, concept-aligned ones

    Args:
        saved_R_path (str): Path to the saved rotation matrix '.bin' file (torch saved tensor).
    """
    # Load the rotation matrix (ensure CPU, detach, convert to NumPy)
    R = torch.load(saved_R_path, weights_only=True).cpu().detach().numpy()

    # Ensure it's a square matrix
    num_dims = R.shape[0]
    assert R.shape[0] == R.shape[1], "Rotation matrix R must be square (NxN)."

    # Extract hypothesis ID; if not found, assume full concept split: i = num_dims // 2.
    hypothesis_id = extract_hypothesis(saved_R_path)
    if hypothesis_id is None:
        hypothesis_id = num_dims // 2
    # Create an AlignmentHypothesis object with h_dim equal to the matrix dimension.
    alignment_hypothesis = AlignmentHypothesis(hypothesis_id, h_dim=num_dims)

    # Initialize arrays to store contributions per original dimension.
    c1_contribution = np.zeros(num_dims)
    c2_contribution = np.zeros(num_dims)
    none_contribution = np.zeros(num_dims)

    # For each original dimension i (i.e., each row of R), compute contributions.
    for i in range(num_dims):
        row_i = R[i, :]
        # Sum of contributions for each concept based on the hypothesis split.
        sum_c1 = np.sum(row_i[alignment_hypothesis.C1_dims]**2) if alignment_hypothesis.C1_dims else 0.0
        sum_c2 = np.sum(row_i[alignment_hypothesis.C2_dims]**2) if alignment_hypothesis.C2_dims else 0.0
        sum_none = np.sum(row_i[alignment_hypothesis.none_dims]**2) if alignment_hypothesis.none_dims else 0.0

        total = sum_c1 + sum_c2 + sum_none + 1e-12  # prevent division by zero

        c1_contribution[i] = sum_c1 / total
        c2_contribution[i] = sum_c2 / total
        none_contribution[i] = sum_none / total

    # Plot a stacked bar chart.
    indices = np.arange(num_dims)
    plt.figure(figsize=(10, 6))

    # Plot bottom layer: C1 contributions (red)
    plt.bar(indices, c1_contribution, color='red', label='C1')
    # Plot middle layer: C2 contributions (blue), stacked on top of C1
    plt.bar(indices, c2_contribution, bottom=c1_contribution, color='blue', label='C2')
    # Plot top layer: None contributions (grey), stacked on top of C1+C2
    plt.bar(indices, none_contribution, bottom=c1_contribution + c2_contribution, color='grey', label='None')

    # Cosmetic adjustments
    plt.xticks(indices, [f"Dim {i}" for i in range(num_dims)], rotation=90)
    plt.ylim([0, 1])
    plt.xlabel("Original Dimension Index", fontsize=12)
    plt.ylabel("Normalized Concept Contribution", fontsize=12)
    plt.title("Concept Contribution per Original Dimension", fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Construct the output path
    img_path = os.path.splitext(saved_R_path)[0] + "_contribution.png"
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved concept contribution bar chart to: {img_path}")


def build_test_R(safe_location):
    # --- Custom Matrix ---
    R_custom = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.1, 0.5],
        [0.0, 0.1, 0.9, 0.0],
        [0.0, 0.5, 0.0, 0.5]
    ])
    # --- Identity ---
    n = 8
    R_identity = np.identity(n)

    # --- Anti-Diagonal Identity ---
    R_anti_identity = np.zeros((n, n), dtype=int)
    for i in range(n // 2):
        R_anti_identity[i, n // 2 + i] = 1  # Upper-right quarter
        R_anti_identity[n // 2 + i, i] = 1  # Lower-left quarter

    # --- Even-Odd Separation Matrix ---
    R_even_odd = np.zeros((n, n), dtype=int)
    indices = [2 * i for i in range(n // 2)] + [2 * i + 1 for i in range(n // 2)]

    # Fill the permutation matrix so that the output position new_idx takes the value from input position old_idx.
    for new_idx, old_idx in enumerate(indices):
        R_even_odd[old_idx, new_idx] = 1

    R = torch.tensor(R_even_odd, dtype=torch.float32)
    torch.save(R, safe_location)


if __name__ == "__main__":
    # build_test_R("trained_models/identity_even_odd_R.bin")
    load_R = "trained_models/mnistnn_MNISTPairsEncoder_R10.bin"
    visualize_rotation_matrix(saved_R_path=load_R)
    visualize_concept_contribution(saved_R_path=load_R)
    # visualize_rotation_degrees(saved_R_path=load_R)
