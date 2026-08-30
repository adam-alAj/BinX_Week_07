# Day 1 — Sprint 2 Kickoff & Convolution

Welcome to Day 1 of Week 7. This session marks the beginning of **Sprint 2** within **Phase 3 — Deep Learning & Applied Project**. We complete Sprint 2 planning, learn the convolution operation through a hands-on edge-detection demonstration, understand why parameter sharing makes CNNs efficient, and confirm that our project's tabular data requires a **dense network** — not a CNN.

---

## 🎯 Objective

- Complete Sprint 2 planning: review Sprint 1 retrospective, define the core-model backlog with acceptance criteria, and commit to the sprint goal
- Apply a hand-defined edge-detection filter to a synthetic image via convolution and visualize the feature map
- Explain why the same filter across the whole image needs far fewer weights than a dense layer (parameter sharing)
- Confirm whether the project's data type calls for a CNN, RNN/Transformer, or the dense network from Week 6, and record the decision

---

## 📓 Notebook

- [Sprint2-Kickoff_Convolution.ipynb](./Sprint2-Kickoff_Convolution.ipynb)

---

## 📋 Sprint 2 Events & Timeline

Phase 3 runs across **4 one-week sprints**. Sprint 2 (Week 7) follows the standard sprint cycle:

| Event | When | Description |
|:------|:-----|:------------|
| **Sprint Planning** | Day 1 | Review Sprint 1 retrospective, define Sprint 2 backlog, commit to the sprint goal. |
| **Daily Stand-up** | Daily | 3-minute update: what was completed, what is next, any blockers. |
| **Mentor Code & Notebook Review** | Day 3 | Mentor reviews notebook and code via GitHub PR with structured comments. |
| **Sprint Review** | Day 5 | Demo completed work. Incomplete tasks move to Sprint 3 with documented reasons. |
| **Sprint Retrospective** | Day 5 | What went well, what to improve, and one specific action for Sprint 3. |

> **Notebook Workflow:** All work is committed to GitHub via **feature branches**. A pull request is opened for mentor review before merging to `main`.

---

## 📊 Sprint 2 Goal

> **Improve the heart-disease classification pipeline through systematic feature engineering,
> cross-validation, and advanced hyperparameter tuning — while correctly selecting the
> architecture that fits the project's tabular data structure.**

### Why Feature Engineering & Cross-Validation?

Sprint 1's retrospective identified that the neural network did not beat the Logistic Regression baseline on any metric:

| Sprint 1 Finding | Sprint 2 Action |
|:-----------------|:----------------|
| NN did not beat baseline (Accuracy: 0.8152 vs 0.8533) | Explore feature engineering to create more informative inputs |
| No cross-validation used | Implement k-fold cross-validation for robust evaluation |
| Narrow tuning space (one-variable-at-a-time) | Expand search with more systematic tuning |
| No learning rate scheduling | Add learning rate schedules (e.g., ReduceLROnPlateau) |
| No structured experiment tracker | Create a consolidated experiment log from Day 1 |

---

## 📋 Sprint 2 Planning — Backlog & Acceptance Criteria

| # | Task | Description | Priority | Acceptance Criteria |
|:-:|:-----|:------------|:--------:|:--------------------|
| 1 | **Feature Engineering** | Create interaction terms, polynomial features, domain-driven transforms | High | New features added; documented rationale; no data leakage |
| 2 | **Cross-Validation Pipeline** | Implement StratifiedKFold (k=5) with preprocessing inside the fold | High | Pipeline runs without leakage; CV scores reported with std |
| 3 | **Expanded Hyperparameter Tuning** | Use GridSearchCV or Optuna for broader search space | High | Search space defined; best params recorded; comparison table produced |
| 4 | **Learning Rate Scheduling** | Add ReduceLROnPlateau or cosine annealing to Keras training | Medium | Loss curves show learning rate adaptation; comparison vs fixed lr |
| 5 | **Experiment Tracker** | Consolidated log of all experiments with timestamps, parameters, metrics | Medium | Single DataFrame/table with all experiments; sorted by performance |
| 6 | **Architecture Exploration** | Test additional architectures beyond [64, 32] (e.g., [128, 64, 32], [256, 128]) | Medium | 3+ architectures compared; best selected with justification |
| 7 | **Sprint 2 Review & Retrospective** | Document results, acceptance criteria, retrospective | Required | All acceptance criteria evaluated; retrospective written |

### Core-Model Objective

Build a neural network (or enhanced classical model) that **beats the Sprint 1 baseline**:

| Metric | Sprint 1 Baseline | Sprint 2 Target |
|:-------|:------------------|:----------------|
| Accuracy | 0.8533 | > 0.8533 |
| F1-Score | 0.8657 | > 0.8657 |
| ROC-AUC | 0.9159 | > 0.9159 |

---

## ✅ Key Tasks & Accomplishments

- **Sprint 2 Planning:** Completed Sprint 2 planning with a 7-task backlog, each with written acceptance criteria. Defined the core-model objective (beat Sprint 1 baseline on Accuracy, F1, and ROC-AUC) and documented the Sprint 1 → Sprint 2 continuity including a full Sprint 1 summary and retrospective action items.

- **Architecture Selection:** Compared the three main deep learning architectures (CNN for images, RNN/Transformer for sequences, Dense for tabular) and concluded that the Heart Disease dataset (918 patients × 14 columns, no spatial grid, no temporal sequence) is **tabular** — making the **dense (fully connected) network** the correct choice. Convolution demonstrated on Day 1 is an educational concept, not a directive to use CNNs.

- **Convolution Theory:** Explained convolution as a mathematical operation where a small filter/kernel slides across a larger input, computing a dot product at each position. Defined key terminology: filter/kernel, feature map, stride, padding, parameter sharing, local connectivity, translation invariance, and feature hierarchy. Explained why understanding hand-designed edge-detection filters builds intuition for what learned CNN filters discover.

- **Edge-Detection Demonstration:** Created a 256×256 synthetic grayscale image with geometric shapes (rectangles, lines, gradients). Defined two hand-crafted 3×3 edge-detection kernels — **horizontal** (detects top-to-bottom intensity changes) and **vertical** (detects left-to-right intensity changes). Applied both via `scipy.ndimage.convolve` and computed a gradient magnitude map combining both directions.

- **Feature Map Visualization:** Visualized the original image, horizontal edge map, vertical edge map, and combined gradient magnitude in a 2×3 subplot grid. Interpreted results: horizontal map highlights horizontal boundaries, vertical map highlights vertical boundaries, and the combined magnitude captures all edge orientations.

- **Parameter Efficiency Analysis:** Demonstrated numerically why convolution is parameter-efficient. A 3×3 kernel with **9 parameters** produces a feature map across 65,025 output positions (256×256 image), while a dense layer connecting the same input to 65,025 outputs would require **4.2 billion parameters** (65,025 × 65,025). This ~470 million× difference comes from **parameter sharing** — the same kernel weights are reused at every spatial position.

---

## 🛠️ Skills Covered

- Sprint planning with backlog, acceptance criteria, and Definition of Done
- Sprint 1 → Sprint 2 continuity and retrospective follow-through
- Architecture selection based on data type (tabular → dense, images → CNN, sequences → RNN)
- Convolution operation: filter/kernel, feature map, stride, padding
- Parameter sharing and local connectivity
- Edge detection with hand-crafted kernels (horizontal and vertical)
- `scipy.ndimage.convolve` for applying convolution to images
- Gradient magnitude computation
- Dense-vs-convolutional parameter comparison
- Synthetic image generation for demonstration
- Matplotlib multi-panel visualization (GridSpec)

---

## 🔗 Related

- [Week 7 Overview](../README.md)
- [Root Repository README](../../README.md)
