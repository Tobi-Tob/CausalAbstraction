import os
import torch
import random
import numpy as np
from pyvene import CausalModel

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


class CausalEqualityModel(CausalModel):
    """
    This model is a simple tree-structured algorithm to solve the hierarchical equality task (Premack 1983).

    The task is defined as:
    - Input: Two pairs of objects.
    - Output: True if both pairs contain the same object or if both pairs contain different objects, False otherwise.
    - Example: AABB and ABCD are both labeled True, while ABCC and BBCD are both labeled False.

    This algorithm solves this task by applying a simple equality relation three times:
    Compute whether the first two inputs are equal, compute whether the second two inputs are equal,
    then compute whether the truth-valued outputs of these first two computations are equal.
    (see https://github.com/stanfordnlp/pyvene/blob/main/tutorials/advanced_tutorials/DAS_Main_Introduction.ipynb)
    """

    def __init__(self, object_dim=2):
        self.object_dim = object_dim

        variables = ["W", "X", "Y", "Z", "WX", "YZ", "O"]

        values = {variable: f"Vectors of size {object_dim} in [-1, 1]" for variable in ["W", "X", "Y", "Z"]}
        values["WX"] = [True, False]
        values["YZ"] = [True, False]
        values["O"] = [True, False]

        parents = {
            "W": [],
            "X": [],
            "Y": [],
            "Z": [],
            "WX": ["W", "X"],
            "YZ": ["Y", "Z"],
            "O": ["WX", "YZ"],
        }

        functions = {
            "W": lambda: np.zeros(self.object_dim),  # Default filler for W
            "X": lambda: np.zeros(self.object_dim),  # Default filler for X
            "Y": lambda: np.ones(self.object_dim),  # Default filler for Y
            "Z": lambda: np.zeros(self.object_dim),  # Default filler for Z
            "WX": lambda x, y: np.array_equal(x, y),
            "YZ": lambda x, y: np.array_equal(x, y),
            "O": lambda x, y: x == y,
        }

        pos = {
            "W": (0.6, 0),
            "X": (1.2, 0.3),
            "Y": (1.8, 0.3),
            "Z": (2.4, 0),
            "WX": (1, 2),
            "YZ": (2, 2),
            "O": (1.5, 3),
        }

        equiv_classes = {}  # What's that?
        super().__init__(variables, values, parents, functions, pos=pos)  # build Causal Model

    def generate_data(self, dim=2, n_examples=131072, include_counterfactuals=False, save_dir="data"):
        def input_sampler():
            A = randvec(dim)
            B = randvec(dim)
            C = randvec(dim)
            D = randvec(dim)
            x = random.randint(1, 4)
            if x == 1:
                return {"W": A, "X": B, "Y": C, "Z": D}
            elif x == 2:
                return {"W": A, "X": A, "Y": B, "Z": B}
            elif x == 3:
                return {"W": A, "X": A, "Y": C, "Z": D}
            elif x == 4:
                return {"W": A, "X": B, "Y": C, "Z": C}

        os.makedirs(save_dir, exist_ok=True)
        print(f"Generating {n_examples} examples in directory '{save_dir}' (ETC: {n_examples / 4000:.0f}s) ...")
        if include_counterfactuals:
            filename = f"equality_task_counterfactual_data_dim{dim}.pt"
            raise NotImplementedError("Counterfactuals are not yet implemented")
        else:
            dataset = self.generate_factual_dataset(n_examples, input_sampler)
            filename = f"equality_task_data_dim{dim}.pt"

        filepath = os.path.join(save_dir, filename)
        torch.save(dataset, filepath)
        print(f"Dataset saved to {filepath}")


def randvec(dim, lower=-1, upper=1):
    return np.array([round(random.uniform(lower, upper), 2) for i in range(dim)])


if __name__ == "__main__":
    object_dim = 2
    # number_of_examples = 20
    # reps = [randvec(object_dim, lower=-1, upper=1) for _ in range(number_of_examples)]

    equality_model = CausalEqualityModel(object_dim)

    equality_model.generate_data(dim=object_dim, n_examples=131072, include_counterfactuals=False)

    #dataset = torch.load("data/equality_task_small_data_dim2.pt", weights_only=True)
    #X = torch.stack([example['input_ids'] for example in dataset])
    #y = torch.stack([example['labels'] for example in dataset])
