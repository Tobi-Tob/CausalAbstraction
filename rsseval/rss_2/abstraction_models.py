import os
from types import SimpleNamespace
import torch
import random
import numpy as np
from pyvene import CausalModel
from datasets.utils.mnist_creation import load_2MNIST

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
        os.makedirs(save_dir, exist_ok=True)

        # Generate counterfactual dataset
        if include_counterfactuals:
            def intervention_id(intervention):
                if "WX" in intervention and "YZ" in intervention:
                    return 2
                if "WX" in intervention:
                    return 0
                if "YZ" in intervention:
                    return 1
                else:
                    raise ValueError("Invalid intervention")

            def input_sampler(*args, **kwargs):
                """
                Returns a potential input {"W": A, "X": B, "Y": C, "Z": D} that constraints output_var to be output_var_value
                """
                A = randvec(dim)
                B = randvec(dim)
                C = randvec(dim)
                D = randvec(dim)
                if kwargs.get('output_var', None) is None:
                    return random.choice([
                        {"W": A, "X": B, "Y": C, "Z": D},
                        {"W": A, "X": A, "Y": B, "Z": B},
                        {"W": A, "X": A, "Y": C, "Z": D},
                        {"W": A, "X": B, "Y": C, "Z": C}
                    ])
                elif kwargs['output_var'] == 'WX' and kwargs['output_var_value']:
                    return random.choice([  # If through intervention WX is set to True
                        {"W": A, "X": A, "Y": C, "Z": D},
                        {"W": A, "X": A, "Y": C, "Z": C}
                    ])
                elif kwargs['output_var'] == 'WX' and not kwargs['output_var_value']:
                    return random.choice([  # If through intervention WX is set to False
                        {"W": A, "X": B, "Y": C, "Z": D},
                        {"W": A, "X": B, "Y": C, "Z": C}
                    ])
                elif kwargs['output_var'] == 'YZ' and kwargs['output_var_value']:
                    return random.choice([  # If through intervention YZ is set to True
                        {"W": A, "X": B, "Y": C, "Z": C},
                        {"W": A, "X": A, "Y": C, "Z": C}
                    ])
                else:
                    return random.choice([  # If through intervention YZ is set to False
                        {"W": A, "X": B, "Y": C, "Z": D},
                        {"W": A, "X": A, "Y": C, "Z": D}
                    ])

            print(f"Generating {n_examples} examples in directory '{save_dir}' (ETC: {n_examples / 1400:.0f}s) ...")

            dataset = self.generate_counterfactual_dataset(
                n_examples, intervention_id, batch_size=6400, sampler=input_sampler
                # what is sampler=self.sample_input_tree_balanced?
            )
            filename = f"equality_task_counterfactual_data_dim{dim}.pt"

        # Generate factual dataset
        else:
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

            print(f"Generating {n_examples} examples in directory '{save_dir}' (ETC: {n_examples / 4000:.0f}s) ...")

            dataset = self.generate_factual_dataset(n_examples, input_sampler)
            filename = f"equality_task_data_dim{dim}.pt"

        filepath = os.path.join(save_dir, filename)
        torch.save(dataset, filepath)
        print(f"Dataset saved to {filepath}")


def randvec(dim, lower=-1, upper=1):
    return np.array([round(random.uniform(lower, upper), 2) for i in range(dim)])


class CausalMnistAdditionModel(CausalModel):
    """
    This model represents a causal abstraction for the MNIST addition task that follows the human reasoning process to solve this task.
    It takes the two digit inputs D1, D2 and uses their ground truth labels C1, C2 (skipping the perceptual part).
    C1 and C2 are then summed to produce the final output SUM.
    - D: Index of a datapoint containing 2 digit images (28x28 MNIST images)
    - C1, C2: Ground truth digit labels (0-9)
    - SUM: Computed sum of C1 and C2
    """

    def __init__(self, data_split="train"):
        variables = ["D", "C1", "C2", "SUM"]
        values = {
            "D": "index of two 28x28 images",
            "C1": list(range(10)),  # Digit ground truth concepts 0-9
            "C2": list(range(10)),
            "SUM": list(range(19))  # Possible sums: 0-18
        }
        parents = {
            "D": [],
            "C1": ["D"],
            "C2": ["D"],
            "SUM": ["C1", "C2"]
        }
        functions = {
            "D": lambda: self.get_all_indices_with_constraint(c1=1, c2=3)[0],  # Index filler, get first index for label '1', '3'
            "C1": lambda D: self.get_c1[D],  # Lookup ground truth from dataset
            "C2": lambda D: self.get_c2[D],
            "SUM": lambda C1, C2: C1 + C2
        }
        pos = {
            "D": (1, 0),
            "C1": (0.5, 1),
            "C2": (1.5, 1),
            "SUM": (1, 2)
        }

        # Load the dataset based on the split
        args = SimpleNamespace(task="addition")
        if data_split == 'train':
            dataset, _, _ = load_2MNIST(args=args)  # Load MNIST Addition dataset args.task == "addition" is required
        elif data_split == 'val':
            _, dataset, _ = load_2MNIST(args=args)
        elif data_split == 'test':
            _, _, dataset = load_2MNIST(args=args)
        else:
            raise ValueError(f"Invalid split: {data_split}")

        self.data_split = data_split
        self.get_c1 = dataset.real_concepts[:, 0]  # Skipping the image perception part we just look at the ground truth labels
        self.get_c2 = dataset.real_concepts[:, 1]
        self.dataset_max_index = len(self.get_c1) - 1

        # Count the number of occurrences of each concept
        c1_c2_counts = {(c1, c2): len(self.get_all_indices_with_constraint(c1, c2)) for c1 in range(3) for c2 in range(3)}
        # print('Occurrences of (C1, C2):', c1_c2_counts)

        super().__init__(variables, values, parents, functions, pos=pos)

    def get_all_indices_with_constraint(self, c1=None, c2=None):
        """
        This method returns all indices of the dataset that satisfy the constraint C1 == c1 and C2 == c2.
        If c1 or c2 is None, it relaxes this constraint.
        Args:
            c1: The value of C1 to filter by.
            c2: The value of C2 to filter by.
        Returns:
            list: A list of indices that satisfy the constraint.
        """
        if c1 is None and c2 is None:
            return list(range(self.dataset_max_index + 1))
        elif c1 is None:
            return np.where(self.get_c2 == c2)[0].tolist()
        elif c2 is None:
            return np.where(self.get_c1 == c1)[0].tolist()
        else:
            return np.where((self.get_c1 == c1) & (self.get_c2 == c2))[0].tolist()

    def generate_data(self, n_examples=1000, batch_size=5, include_counterfactuals=False, save_dir="data"):
        os.makedirs(save_dir, exist_ok=True)

        def input_sampler(output_var=None, output_var_value=None):
            """
            Returns a potential input index {"D": x} such that the digit images suit the intervention.
            Sample an index such that the constraint output_var (C1 or C2) is equal to output_var_value.
            """
            if output_var is None:  # Randomly sample a datapoint from the dataset
                index = random.choice(self.get_all_indices_with_constraint(c1=None, c2=None))  # With random C1 and C2
            elif output_var == 'C1':
                index = random.choice(self.get_all_indices_with_constraint(c1=output_var_value, c2=None))  # With fixed C1, random C2
            elif output_var == 'C2':
                index = random.choice(self.get_all_indices_with_constraint(c1=None, c2=output_var_value))  # With random C1, fixed C2
            else:
                raise ValueError(f"Invalid output_var: {output_var}")
            return {"D": index}

        # Generate counterfactual dataset
        if include_counterfactuals:
            def intervention_id(intervention):
                if "C1" in intervention and "C2" in intervention:
                    return 2
                if "C1" in intervention:
                    return 0
                if "C2" in intervention:
                    return 1
                else:
                    raise ValueError("Invalid intervention")

            print(f"Generating {n_examples} counterfactual examples in '{save_dir}' (ETC: {n_examples / 600:.0f}s) ...")

            dataset = self.generate_counterfactual_dataset(n_examples, intervention_id, batch_size=batch_size, sampler=input_sampler)
            filename = f"mnist_add_counterfactual_{self.data_split}_data_bs{batch_size}.pt"

        # Generate factual dataset
        else:
            print(f"Generating {n_examples} factual examples in '{save_dir}' (ETC: {n_examples / 700:.0f}s) ...")
            dataset = self.generate_factual_dataset(n_examples, input_sampler)
            filename = f"mnist_add_{self.data_split}_data.pt"

        filepath = os.path.join(save_dir, filename)
        torch.save(dataset, filepath)
        print(f"Dataset saved to {filepath}")

    def print_example_run(self):
        print("Timesteps:", self.timesteps)
        base_input = {"D": self.get_all_indices_with_constraint(c1=5, c2=3)[0]}
        self.print_setting(self.run_forward(base_input))

        print("Intervention: setting C1 to 9:")
        self.print_setting(self.run_forward({"D": self.get_all_indices_with_constraint(c1=5, c2=3)[0], "C1": 9}))

    def print_dataset_entry(self, dataset, idx):
        print('Base input:', self.get_c1[int(dataset[idx]["input_ids"].tolist()[0])],
              self.get_c2[int(dataset[idx]["input_ids"].tolist()[0])])
        print('Base label:', int(dataset[idx]["base_labels"].tolist()[0]))
        print('Source input A:', self.get_c1[int(dataset[idx]["source_input_ids"].tolist()[0][0])],
              self.get_c2[int(dataset[idx]["source_input_ids"].tolist()[0][0])])
        print('Source input B:', self.get_c1[int(dataset[idx]["source_input_ids"].tolist()[1][0])],
              self.get_c2[int(dataset[idx]["source_input_ids"].tolist()[1][0])])
        print('Intervention ID:', dataset[idx]["intervention_id"].tolist()[0])
        print('Label after intervention:', int(dataset[idx]["labels"].tolist()[0]))


if __name__ == "__main__":
    causal_model = CausalMnistAdditionModel(data_split="train")
    causal_model.generate_data(n_examples=200000, batch_size=20, include_counterfactuals=True)

    # dataset = torch.load("data/mnist_add_counterfactual_train_data_bs100.pt", weights_only=True)
    # causal_model.print_dataset_entry(dataset, 0)




