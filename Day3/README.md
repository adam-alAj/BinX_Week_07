# Day 3 — RNNs & LSTMs for Sequential Data

Welcome to Day 3 of Week 7. This session continues **Sprint 2** within **Phase 3 — Deep Learning & Applied Project**. Completing the mid-sprint architectural trio (CNNs → RNNs → LSTMs), we shift from spatial data (Day 1–2) to **temporal/sequential data** — building a plain RNN baseline and then a stacked bidirectional LSTM to classify ECG heartbeat signals, demonstrating why gated memory matters for long sequences.

Day 3 is also the mid-sprint **Mentor Code & Notebook Review** point.

---

## 🎯 Objective

1. Explain why sequential data requires an order-aware architecture
2. Describe how an RNN's hidden state carries memory across a sequence
3. Explain the vanishing-gradient problem and how LSTMs solve it
4. Build and train a plain RNN baseline on ECG sequential data
5. Build and train a stacked bidirectional LSTM on the same data
6. Compare the LSTM against the plain RNN and interpret the difference
7. Connect the Day 3 sequential models to the CNN work of Days 1 and 2

---

## 📓 Notebook

- [RNNs_LSTMs.ipynb](./RNNs_LSTMs.ipynb)

---

## 🗂️ Dataset

| Property | Value |
|:---------|:------|
| **Task** | 5-class arrhythmia classification (N, S, V, F, Q) |
| **Source** | MIT-BIH Arrhythmia Database ([Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)) |
| **Train samples** | 87,554 heartbeats |
| **Test samples** | 21,892 heartbeats |
| **Sequence length** | 187 timesteps × 1 feature (ECG voltage) |
| **Class imbalance** | ~113× between majority (Class 0: 82.8%) and minority (Class 3: 0.73%) |

### Class Codes

| Code | Label | Meaning |
|:----:|:------|:--------|
| 0 | N | Normal beat |
| 1 | S | Supraventricular ectopic beat |
| 2 | V | Ventricular ectopic beat |
| 3 | F | Fusion of ventricular and normal beat |
| 4 | Q | Unclassifiable beat |

> **Why this dataset?** ECG heartbeat signals are **time-series data** where the order of voltage readings carries clinical meaning. This makes them the correct modality for demonstrating RNN/LSTM architectures — just as skin lesion images (Day 2) were the correct modality for CNNs.

---

## 🏗️ Experiments

### Experiment 1 — Plain RNN Baseline

A single-layer SimpleRNN classifier:

```
Input (187, 1)
↓
SimpleRNN (64 units, tanh, return_sequences=False)
↓
Dropout (0.3)
↓
Dense (5, softmax)
```

- **Parameters:** 4,549
- **Optimizer:** Adam (lr=0.001, gradient clipping via `clipnorm=1.0`)
- **Loss:** `sparse_categorical_crossentropy` with class weights
- **Purpose:** Establish a baseline that demonstrates the vanishing gradient limitation over 187 timesteps

### Experiment 2 — Stacked Bidirectional LSTM

A two-layer bidirectional LSTM with batch normalization:

```
Input (187, 1)
↓
Bidirectional(LSTM(64, return_sequences=True))
↓
BatchNormalization → Dropout (0.3)
↓
Bidirectional(LSTM(32))
↓
BatchNormalization → Dropout (0.3)
↓
Dense (5, softmax)
```

- **Parameters:** 76,101
- **Optimizer:** Adam (lr=1e-3) with `ReduceLROnPlateau` (factor=0.5, patience=3)
- **Callbacks:** EarlyStopping (patience=5, restore_best_weights=True)
- **Key innovation:** Cell state updated via **addition** (not multiplication), creating a "gradient highway" that preserves gradients over all 187 timesteps

### Experiment 3 — Order-Awareness Experiment (Ablation)

The same Stacked BiLSTM trained on **shuffled** sequences (time-steps randomly permuted within each sample). If sequence order matters, the shuffled model should perform worse because temporal information is destroyed.

---

## 📊 Results

### Model Comparison Table

| Model | Parameters | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Precision | Test Recall |
|:------|:-----------|:-------------|:-------------|:--------------|:--------------|:---------------|:------------|
| **Stacked BiLSTM** | 76,101 | 0.9321 | 0.7783 | **0.9275** | **0.7676** | 0.7019 | 0.9176 |
| Plain RNN | 4,549 | 0.7693 | 0.5613 | 0.7701 | 0.5597 | 0.5468 | 0.7336 |
| LSTM (shuffled) | 76,101 | — | — | 0.5816 | 0.4219 | — | — |

### Key Differences

- **Test Accuracy improvement (LSTM − RNN):** +0.1574
- **Test Macro F1 improvement (LSTM − RNN):** +0.2079
- **Shuffling degradation:** Destroying sequence order degraded Macro F1 from 0.7676 → 0.4219, confirming that temporal order contains useful diagnostic information

---

## 📈 Visualizations

1. **ECG signal visualization** — sample heartbeats from each of the 5 classes
2. **Class distribution** — bar chart showing severe imbalance (82.8% Normal vs 0.73% Fusion)
3. **Training curves** — accuracy and loss vs. epoch for both RNN and LSTM
4. **RNN vs LSTM comparison** — side-by-side training curves
5. **Confusion matrices** — per-model confusion matrices on the test set
6. **Classification reports** — precision, recall, F1-score per class for both models
7. **Order-awareness comparison** — ordered vs. shuffled LSTM performance

---

## ✅ Key Tasks & Accomplishments

- **Architecture Selection:** Explained why ECG signals require sequential models (RNN/LSTM) rather than CNNs (spatial) or dense networks (tabular), completing the Week 7 architectural trio (Day 1: convolution concepts, Day 2: CNNs, Day 3: RNNs/LSTMs).

- **Plain RNN Baseline:** Built and trained a single-layer SimpleRNN (64 units) with dropout and gradient clipping. Achieved Test Accuracy 0.7701 and Macro F1 0.5597 — demonstrating the vanishing gradient limitation over 187 timesteps.

- **Stacked Bidirectional LSTM:** Built and trained a two-layer bidirectional LSTM (64 → 32 units) with batch normalization, dropout, learning rate scheduling, and early stopping. Achieved Test Accuracy 0.9275 and Macro F1 0.7676 — a +20.79% improvement in Macro F1 over the plain RNN.

- **Vanishing Gradient Demonstration:** The comparison empirically validates that LSTM's gated cell state (forget, input, output gates) solves the vanishing gradient problem that limits plain RNNs over long sequences.

- **Order-Awareness Experiment:** Shuffling time-steps within each sample degraded LSTM Macro F1 from 0.7676 → 0.4219, providing empirical evidence that temporal ordering contains clinically meaningful information.

- **Class Imbalance Handling:** Applied `compute_class_weight('balanced')` to handle the ~113× imbalance between majority and minority classes. Macro F1 was used as the primary metric (accuracy is misleading at 82.8% majority class).

- **Mentor Review Ready:** Notebook is fully documented with clear markdown narratives, reproducible results (seed=42), and a comprehensive RNN vs. LSTM comparison.

---

## 🛠️ Skills Covered

- **SimpleRNN:** Hidden state as memory, weight matrices, tanh activation
- **Vanishing Gradient Problem:** Why gradients shrink over long sequences
- **LSTM Gates:** Forget gate, input gate, output gate, cell state
- **Cell State:** Additive updates creating a "gradient highway"
- **Bidirectional Processing:** Forward and backward sequence passes
- **Stacked LSTMs:** Multi-layer temporal abstraction
- **BatchNormalization:** Stabilizing LSTM training
- **Gradient Clipping:** `clipnorm=1.0` for RNN training stability
- **Class Weights:** `compute_class_weight('balanced')` for imbalanced data
- **EarlyStopping & ReduceLROnPlateau:** Training callbacks
- **Macro F1-Score:** Primary metric for imbalanced multi-class problems
- **Ablation Studies:** Order-awareness experiment (shuffled vs. ordered)
- **TensorFlow/Keras:** Sequential API, SimpleRNN, LSTM, Bidirectional layers

---

## 🔗 Related

- [Day 1 — Sprint 2 Kickoff & Convolution](../Day1/README.md)
- [Day 2 — Building CNNs & Transfer Learning](../Day2/README.md)
- [Week 7 Overview](../README.md)
- [Root Repository README](../../README.md)
