# Reverse-Engineering the Internal Reasoning of Neuro-Symbolic AI: Where Are the Concepts?

![Python](https://img.shields.io/badge/python-3.8-green.svg)
[![Paper](https://img.shields.io/badge/Read-Project_Report-blue.svg)](./Causal_Abstraction_of_NeSy_models.pdf)

## 💡 Why This Project Matters (Intuitive Analogy) 

Imagine a student who scores **100%** on a math test. 

You assume they understand arithmetic. However, what if they did not actually learn how to add? What if they simply noticed that whenever a question has a blue background, the correct answer is "7"? 

This is a **Reasoning Shortcut (RS)** [1]. In real-world applications—such as healthcare or autonomous driving—this kind of internal "cheating" can lead to catastrophic out-of-distribution failures. 

Standard AI evaluations only look at the final test accuracy. **This project is different.** We go beyond performance metrics by performing "neural brain surgery" to map the model's inner reasoning. Using mathematical frameworks (**Causal Abstraction Theory** [2]), we verify whether human-interpretable concepts actually exist inside the neural network's latent space, or if the model is merely exploiting a shortcut.

---

## 🚀 Scientific Overview

Understanding how neural models encode concepts internally is a central challenge in explainable AI (XAI) and mechanistic interpretability. While **Neuro-Symbolic (NeSy)** models are designed to improve interpretability by combining neural perception with symbolic logic, they remain susceptible to reasoning shortcuts instead of learning meaningful representations. 

In this work, we analyze this phenomenon in **DeepProbLog (DPL)** models applied to the **MNIST-Addition** task, which requires both visual perception and reasoning. Using **Causal Abstraction theory** [2] and **Distributed Alignment Search (DAS)** [3], we investigate whether these models can be described by a high-level interpretable reasoning process and precisely where they encode abstract concepts. Our findings reveal that architectural choices strongly influence the reliability of internal concept encodings, offering concrete insights into which reasoning shortcuts occur and how abstract concept learning can be improved in hybrid models.

---

## 📌 Key Takeaways

*   ✅ **DeepProbLog models are not immune to reasoning shortcuts (RS):** High downstream accuracy can mask severe internal representation failures.
*   ✅ **Identification of concept flipping and collapse:** We observe these failure modes as unintended local optima of the learning objective, where models leverage spurious correlations while still achieving high task accuracy.
*   ✅ **Disentanglement as a mitigation strategy:** Enforcing structural disentanglement in the concept encoder significantly helps to isolate representations and mitigate reasoning shortcuts.
*   ✅ **Localized concept encoding:** Concept information is not evenly distributed across latent dimensions; our alignment search shows it plateaus within specific, localized subsets of the latent space.

For a detailed explanation of our methodology, causal graphs, and empirical findings, please refer to our full project report: [Causal_Abstraction_of_NeSy_models.pdf](./Causal_Abstraction_of_NeSy_models.pdf).

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

## 🔗 References & Acknowledgments
This project is built as an extension of the [RSBench codebase](https://github.com/unitn-sml/rsbench-code) for investigating reasoning shortcuts.

***
*   **[1]** Marconato, E., et al. "Neuro-symbolic AI is not immune to reasoning shortcuts." *arXiv preprint arXiv:2310.15049 (2023)* [1].
*   **[2]** Geiger, A., Lu, H., Icard, T., & Potts, C. "Causal Abstractions of Neural Networks." *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*.
*   **[3]** Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. D. "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations." *arXiv preprint arXiv:2303.02536 (2023)*.
