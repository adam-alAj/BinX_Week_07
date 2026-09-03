# Day 5 — Sprint 2 Close-Out: Advancing the Core Model & Sprint Review

**Week 7 · Phase 3 Sprint 2 — CNNs, RNNs & Transformers**

---

## Objective

Complete the Sprint 2 close-out by:

1. Confirming and justifying the core architecture based on the project's tabular data type.
2. Training and tuning the improved core model through systematic experimentation.
3. Comparing the Sprint 2 model against the Week 6 baseline and Sprint 1 network.
4. Documenting the Sprint Review and Sprint Retrospective.

---

## Dataset & Task

| Property | Value |
|:---------|:------|
| **Dataset** | Merged UCI Heart Disease (Cleveland, Hungary, Switzerland, VA Long Beach) |
| **Samples** | 918 patients |
| **Features** | 11 raw clinical features → **15 preprocessed** after OneHotEncoding |
| **Target** | Binary classification (0 = no disease, 1 = heart disease) |
| **Class balance** | 55.3% positive / 44.7% negative |
| **Train/Test split** | 80/20, stratified, random_state=42 |
| **Primary metric** | F1-Score (binary, positive class) |

---

## Architecture Decision

**Selected: Dense (Fully Connected) Network** — the correct architecture for tabular clinical data per the Week 7 principle: *"Match the architecture to the data and task."*

| Architecture | Why Rejected |
|:-------------|:-------------|
| CNN | Images require spatial structure; our data is tabular |
| LSTM / RNN | Sequential data requires ordered input; our features are independent |
| Transformer | Text/NLP data; our task is tabular classification |

The Day 3 LSTM (ECG signals, F1=0.7676) and Day 4 Transformer (Arabic text, F1=0.9000) were educational experiments on different datasets, not candidates for the project's core model.

---

## Experiments

6 systematic experiments were logged:

| Experiment | Architecture | Dropout | L2 | Scheduler | F1 | Accuracy |
|:-----------|:-------------|:--------|:---|:----------|---:|---------:|
| EXP-00 | [64, 32] | 0.5 | 0.0 | None | 0.8517 | 0.8315 |
| EXP-01 | [64, 32] | 0.5 | 0.0 | ReduceLROnPlateau | 0.8421 | 0.8207 |
| **EXP-02** | **[128, 64]** | **0.5** | **0.0** | **ReduceLROnPlateau** | **0.8585** | **0.8370** |
| EXP-03 | [128, 64, 32] | 0.5 | 0.0 | ReduceLROnPlateau | 0.8374 | 0.8207 |
| EXP-04 | [128, 64] | 0.3 | 0.001 | ReduceLROnPlateau | 0.8402 | 0.8098 |
| EXP-05 (Final) | [128, 64] | 0.3 | 0.001 | ReduceLROnPlateau | 0.8426 | 0.8315 |

**Best experiment by F1: EXP-02 (F1=0.8585)**
**Final evaluated model: EXP-05 (F1=0.8426)**

---

## Results

### Final Comparison

| Model | Accuracy | F1-Score | ROC-AUC |
|:------|:---------|:---------|:--------|
| Week 6 Baseline (Logistic Regression) | 0.7935 | 0.8208 | 0.8940 |
| Random Forest (Week 6) | 0.8315 | 0.8517 | 0.8812 |
| Sprint 1 DL v3 (Tuned) | 0.8478 | 0.8654 | 0.9093 |
| **Sprint 2 Final (EXP-05)** | **0.8315** | **0.8426** | **0.9041** |

### Improvement Over Week 6 Baseline

- **F1-Score:** 0.8208 → 0.8426 (+0.0218, **+2.66%**)
- **Beat the baseline: YES ✓**

### Improvement Over Sprint 1

- **F1-Score:** 0.8654 → 0.8426 (-0.0228, -2.63%)
- **Beat Sprint 1: NO ✗**

Sprint 2 achieved a measurable improvement over the Week 6 baseline but did not surpass the Sprint 1 neural network.

---

## Key Findings

1. **Architecture-to-data matching works**: The dense network is the correct choice for tabular clinical data.
2. **LR scheduling stabilizes training**: ReduceLROnPlateau improved convergence across all experiments.
3. **Wider architecture helps**: [128, 64] outperformed [64, 32] in the best experiment (EXP-02).
4. **Deeper is not always better**: [128, 64, 32] underperformed [128, 64] on this small dataset.
5. **Regularization balance matters**: Combined Dropout + L2 provided stable training but not the highest F1.

---

## Sprint Review & Retrospective

- **Sprint 2 Goal:** Advance the core model to beat the Week 6 baseline — **Achieved ✓**
- **Sprint 3 Action:** Implement k-fold cross-validation and explore gradient boosting (XGBoost/LightGBM).

---

## Notebook

📄 [`Sprint2_Close-Out.ipynb`](./Sprint2_Close-Out.ipynb)

---

## Skills Covered

| Skill | Application |
|:------|:------------|
| Architecture Selection | Dense network for tabular data |
| Experiment Tracking | 6 logged experiments with full config |
| Learning Rate Scheduling | ReduceLROnPlateau |
| Regularization | Dropout + L2 combination |
| Model Evaluation | Accuracy, F1, Precision, Recall, ROC-AUC |
| Confusion Matrix | TN/FP/FN/TP breakdown |
| Baseline Comparison | Week 6 LR vs Sprint 1 vs Sprint 2 |
| Sprint Review | Deliverables, metrics, improvement analysis |
| Sprint Retrospective | What went well, what to improve, Sprint 3 action |
