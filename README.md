# Causal Abstractions of NeSy models. Where are the concepts?
![Python](https://img.shields.io/badge/python-3.8-green.svg)
## 🚀 Overview
 Understanding how neural models encode concepts internally is a central challenge in explainable AI. While Neuro-Symbolic (NeSy) models are designed to improve interpretability, they can still rely on reasoning shortcuts rather than learning meaningful abstractions. In this work, we analyze this phenomenon in DeepProbLog (DPL) models applied to the MNIST-Addition task, which requires both visual perception and reasoning. Using Causal Abstraction theory and Distributed Alignment Search (DAS), we investigate whether these models can be described by a high-level interpretable reasoning process and where they encode abstract concepts. Our findings reveal that architectural choices strongly influence the reliability of internal concept encodings, offering insights into which reasoning shortcuts may occur and into how abstract concept learning can be improved in NeSy models.

## 📌 Key Takeaways
✅ **DeepProbLog models are not immune to reasoning shortcuts (RS).**

✅ **We observe concept flipping and collapse as RS** (unintended optima of the learning objective where models
 achieve high accuracy while leveraging spurious correlations rather than meaningful abstractions).

✅ **Enforcing Disentanglement in the concept Encoder** helps to mitigate RS.

✅ **Concept information is not evenly distributed across latent dimensions**.

For a detailed explanation of our methodology and findings, please refer to our full project report: [Causal_Abstraction.pdf](./Causal_Abstraction.pdf).

---
## 📂 Directory Structure
```
.
├── backbones/             # Definition of the image encoders
├── data/                  # Directory to save the data e.g. MNIST
├── datasets/              # Code for the datasets
├── models/                # Definition of the model architecture
├── trained_models/        # Saved trained models
├── utils/                 # Additional helper functionality 
│
├── abstraction_models.py  # Definition of the causal abstraction that is used
├── DAS.py                 # Code to run distributed alignment search
├── main.py                # Training for different models to solve the task
├── visualizer.py          # Visualization of model behavior and DAS alignment
├── wrapped_models.py      # Code to wrap models to work with pyvene and the DAS implementation
│
├── Causal_Abstraction.pdf # Project report summary
└── environment.yml        # Required dependencies
```

---
This project is an extension on [this](https://github.com/unitn-sml/rsbench-code) implementation of reasoning shortcuts
