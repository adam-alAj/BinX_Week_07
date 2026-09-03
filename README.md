# Week 7: Deep Learning & Applied Project — Sprint 2 (Convolution & Pipeline Improvement)

**Welcome to Week 7 of the BinX Tech AI & Machine Learning Internship Program.**
This week is **Sprint 2** of **Phase 3 — Deep Learning & Applied Project**. Sprint 1 (Week 6) established a robust, reproducible ML baseline pipeline on the Heart Disease dataset. Sprint 2 focuses on understanding convolution and the architecture-selection principle, then improving the classification pipeline through feature engineering, cross-validation, and systematic hyperparameter tuning.

---

## 📋 Table of Contents

| Day | Topic | Notebook | Status |
|:---:|-------|----------|:------:|
| 1 | Sprint 2 Kickoff & Convolution | [`Sprint2-Kickoff_Convolution.ipynb`](./Day1/Sprint2-Kickoff_Convolution.ipynb) | ✅ |
| 2 | Building CNNs & Transfer Learning | [`Building_CNNs.ipynb`](./Day2/Building_CNNs.ipynb) | ✅ |
| 3 | RNNs & LSTMs for Sequential Data | [`RNNs_LSTMs.ipynb`](./Day3/RNNs_LSTMs.ipynb) | ✅ |
| 4 | Attention & Transformers | [`Attention_Transformers.ipynb`](./Day4/Attention_Transformers.ipynb) | ✅ |
| 5 | Sprint 2 Close-Out & Model Advancement | [`Sprint2_Close-Out.ipynb`](./Day5/Sprint2_Close-Out.ipynb) | ✅ |

---

## 📖 Summary by Day

### [Day 1 — Sprint 2 Kickoff & Convolution](./Day1/README.md)
Completed **Sprint 2 planning** with a 7-task backlog and acceptance criteria, then learned convolution through a hands-on edge-detection demonstration. Reviewed the Sprint 1 retrospective and defined the Sprint 2 goal: improve the heart-disease classification pipeline through feature engineering, cross-validation, and systematic tuning. Compared the three main deep learning architectures (CNN for images, RNN/Transformer for sequences, Dense for tabular) and confirmed that the Heart Disease dataset (918 patients × 14 columns) is **tabular** — making the **dense (fully connected) network** the correct choice. Created a 256×256 synthetic grayscale image with geometric shapes and applied two hand-crafted 3×3 edge-detection kernels (horizontal and vertical) via `scipy.ndimage.convolve`. Visualized the original image, both feature maps, and the combined gradient magnitude in a 2×3 subplot grid. Demonstrated numerically why convolution is parameter-efficient: a 3×3 kernel with **9 parameters** produces a feature map across 65,025 output positions, while a dense layer would require **4.2 billion parameters** — a ~470 million× difference from parameter sharing. The notebook also defines the Sprint 2 backlog (7 tasks with priorities and acceptance criteria), core-model targets (beat Accuracy > 0.8533, F1 > 0.8657, ROC-AUC > 0.9159), and the full sprint events timeline.

### [Day 2 — Building CNNs & Transfer Learning](./Day2/README.md)
Completed the required computer-vision educational lab using a **skin lesion image dataset** (11,879 train + 2,000 test images, binary Benign vs Malignant classification). Built and compared three approaches: (1) **CNN From Scratch** — a 3-block Conv2D→MaxPool network trained on 128×128 images with ~4.3M parameters; (2) **CNN + Data Augmentation** — same architecture with RandomFlip, RandomRotation, and RandomZoom applied during training via the Functional API; (3) **Transfer Learning** — frozen MobileNetV2 backbone (ImageNet pre-trained) with a new GlobalAveragePooling2D→Dense classification head trained on 224×224 images. All three experiments use identical train/val/test splits, Adam optimizer, binary cross-entropy loss, and EarlyStopping. Includes training curves, baseline-vs-augmentation comparison, three-way experiment table, confusion matrix, and classification report. Clearly documented that this CNN lab is an educational exercise — the project's core model remains the **dense network from Week 6** for the tabular Heart Disease dataset. Dataset: [Melanoma Skin Cancer — Benign vs Malignant](https://www.kaggle.com/datasets/ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant) (Kaggle).

### [Day 3 — RNNs & LSTMs for Sequential Data](./Day3/README.md)
Completed the mid-sprint architectural trio by shifting from spatial data (Days 1–2) to **temporal/sequential data** using ECG heartbeat signals from the MIT-BIH Arrhythmia Database (87,554 train + 21,892 test heartbeats, 187 timesteps, 5-class arrhythmia classification with ~113× class imbalance). Built and compared three experiments: (1) **Plain SimpleRNN Baseline** — a single-layer SimpleRNN (64 units) with dropout and gradient clipping achieving Test Accuracy 0.7701 and Macro F1 0.5597, demonstrating the vanishing gradient limitation over 187 timesteps; (2) **Stacked Bidirectional LSTM** — a two-layer bidirectional LSTM (64 → 32 units) with batch normalization, learning rate scheduling, and early stopping achieving Test Accuracy 0.9275 and Macro F1 0.7676 (+20.79% Macro F1 improvement), validating that LSTM's gated cell state solves the vanishing gradient problem; (3) **Order-Awareness Ablation** — the same LSTM trained on shuffled sequences degraded Macro F1 from 0.7676 → 0.4219, confirming temporal order contains clinically meaningful information. Used `compute_class_weight('balanced')` for the severe imbalance and Macro F1 as the primary metric. Also serves as the mid-sprint **Mentor Code & Notebook Review** point.

### [Day 4 — Attention & Transformers](./Day4/README.md)
Completed the Week 7 architectural quartet by shifting from sequential signal processing (Day 3) to **natural language processing** — fine-tuning a pre-trained **AraBERT v2** Transformer on the **330K Arabic Sentiment Reviews** dataset (binary positive/negative classification). Explained the fundamental limitations of RNN/LSTM memory (sequential bottleneck, vanishing gradients, fixed-length hidden state) and how **self-attention** solves them (parallel processing, direct long-range connections, dynamic relevance). Described the full Transformer architecture — token embeddings, positional encoding, multi-head self-attention, feed-forward networks, layer normalization, and residual connections. Loaded AraBERT v2 (`aubmindlab/bert-base-arabertv2`, ~110M parameters) pre-trained on 67M Arabic sentences, created a stratified 20,000-sample subset with 50/50 class balance, and fine-tuned using the Hugging Face `Trainer` API with FP16 mixed precision, linear warmup+decay scheduling, and a custom `EarlyStoppingTrainer` subclass. Compared the Transformer against the Day 3 LSTM across architectural properties (parallelization, long-range dependencies, pre-trained knowledge, data efficiency, interpretability, inference speed) and made the **evidence-based architecture decision** to use AraBERT as the project's core architecture for Arabic text classification — following the Week 7 principle: match the architecture to the data type (Text → Transformer, Signals → LSTM, Images → CNN). Also refactored the notebook from a CPU-only manual-loop training script into a GPU-accelerated Colab-ready pipeline with automatic checkpoint persistence to Google Drive.

### [Day 5 — Sprint 2 Close-Out & Model Advancement](./Day5/README.md)
Completed the Sprint 2 close-out by confirming the **dense (fully connected) network** as the correct core architecture for the project's tabular clinical data (918 patients × 11 features), following the Week 7 principle: *match the architecture to the data and task*. Rejected CNN (images), LSTM (sequential signals), and Transformer (text) as inappropriate for tabular data — the Day 3 LSTM (ECG, F1=0.7676) and Day 4 AraBERT (Arabic text, F1=0.9000) were educational experiments on different datasets. Conducted 6 systematic experiments (EXP-00 to EXP-05) with full configuration logging, testing wider/deeper architectures, learning rate scheduling (ReduceLROnPlateau), and combined Dropout+L2 regularization. **EXP-02** (wider [128,64] + LR scheduling) achieved the highest F1=0.8585. The **final evaluated model EXP-05** ([128,64] + Dropout(0.3) + L2(0.001) + scheduling) achieved F1=0.8426, beating the Week 6 Logistic Regression baseline (F1=0.8208) by +2.66% but not surpassing the Sprint 1 neural network (F1=0.8654). Produced a complete comparison table, confusion matrix, classification report, training/validation curves, Sprint Review evidence, and Sprint Retrospective with a concrete Sprint 3 action (k-fold cross-validation + gradient boosting exploration).

---

## 🛠️ Skills & Tools Covered

| Skill | Day | Application |
|-------|:---:|-------------|
| **Sprint Planning** | 1 | Backlog, acceptance criteria, Definition of Done, sprint timeline |
| **Sprint 1 → Sprint 2 Continuity** | 1 | Retrospective follow-through, action items carried forward |
| **Architecture Selection** | 1 | CNN vs RNN vs Dense — match architecture to data type |
| **Convolution Operation** | 1 | Filter/kernel, feature map, stride, padding |
| **Parameter Sharing** | 1 | Same kernel weights reused at every spatial position |
| **Local Connectivity** | 1 | Each output neuron connects to a small local region |
| **Edge Detection** | 1 | Hand-crafted horizontal and vertical 3×3 kernels |
| **`scipy.ndimage.convolve`** | 1 | Applying convolution to 2D images |
| **Gradient Magnitude** | 1 | Combining horizontal and vertical edge maps |
| **Dense-vs-Convolutional Parameters** | 1 | Numerical comparison: 9 params vs 4.2 billion |
| **Synthetic Image Generation** | 1 | Creating geometric shapes for demonstration |
| **Matplotlib GridSpec** | 1 | Multi-panel visualization layouts |
| **CNN Architecture (Conv2D + MaxPool)** | 2 | 3-block CNN from scratch with learned filters |
| **Data Augmentation** | 2 | RandomFlip, RandomRotation, RandomZoom as regularization |
| **Transfer Learning (MobileNetV2)** | 2 | Frozen pre-trained backbone + new classification head |
| **Functional API** | 2 | Composing nested Sequential models correctly |
| **tf.keras Image Data Loading** | 2 | `image_dataset_from_directory` for real image datasets |
| **MobileNetV2 Preprocessing** | 2 | Pixel normalization to [-1, 1] |
| **EarlyStopping Callback** | 2 | Preventing wasteful training epochs |
| **Model Evaluation** | 2 | Accuracy, loss, confusion matrix, classification report |
| **SimpleRNN** | 3 | Hidden state as memory, weight matrices, tanh activation |
| **Vanishing Gradient Problem** | 3 | Why gradients shrink over long sequences |
| **LSTM Gates** | 3 | Forget gate, input gate, output gate, cell state |
| **Cell State** | 3 | Additive updates creating a "gradient highway" |
| **Bidirectional Processing** | 3 | Forward and backward sequence passes |
| **Stacked LSTMs** | 3 | Multi-layer temporal abstraction |
| **BatchNormalization** | 3 | Stabilizing LSTM training |
| **Gradient Clipping** | 3 | clipnorm=1.0 for RNN training stability |
| **Class Weights** | 3 | compute_class_weight('balanced') for imbalanced data |
| **Macro F1-Score** | 3 | Primary metric for imbalanced multi-class problems |
| **Ablation Studies** | 3 | Order-awareness experiment (shuffled vs. ordered) |
| **Attention Mechanism** | 4 | Query-Key-Value vectors, self-attention, multi-head attention |
| **Transformer Architecture** | 4 | Embeddings, positional encoding, feed-forward, layer norm, encoder stack |
| **Pre-trained Transformers (BERT)** | 4 | MLM, NSP, fine-tuning, AraBERT for Arabic NLP |
| **Hugging Face Ecosystem** | 4 | AutoTokenizer, AutoModelForSequenceClassification, Trainer API |
| **FP16 Mixed Precision** | 4 | GPU-accelerated training with reduced memory footprint |
| **Architecture Selection** | 4 | Matching model architecture to data type (Text → Transformer) |
| **TensorFlow/Keras Sequential** | 3 | SimpleRNN, LSTM, Bidirectional layers |
| **Experiment Tracking** | 5 | 6 experiments with full configuration and metrics |
| **Learning Rate Scheduling** | 5 | ReduceLROnPlateau for convergence stability |
| **L2 Regularization** | 5 | Weight decay combined with Dropout |
| **Sprint Review** | 5 | Deliverables, metrics, improvement analysis |
| **Sprint Retrospective** | 5 | What went well, what to improve, Sprint 3 action |

---

## 📁 Folder Structure

```
BinX_Week_07/
├── Day1/
│   ├── Sprint2-Kickoff_Convolution.ipynb
│   └── README.md
├── Day2/
│   ├── Building_CNNs.ipynb
│   └── README.md
├── Day3/
│   ├── RNNs_LSTMs.ipynb
│   └── README.md
├── Day4/
│   ├── Attention_Transformers.ipynb
│   ├── Attention_Transformers1.ipynb
│   └── README.md
├── Day5/
│   ├── Sprint2_Close-Out.ipynb
│   └── README.md
└── README.md                      ← You are here
```

---

## 🚀 How to Run

1. **Clone the parent repository:**
   ```bash
   git clone --recurse-submodules https://github.com/adam-alAj/BinX-ML-Internship.git
   ```

2. **Navigate to Week 7:**
   ```bash
   cd BinX_ML_Internship/BinX_Week_07
   ```

3. **Activate the virtual environment** (located at the parent root):
   ```bash
   ..\\.venv\\Scripts\\activate        # Windows
   source ../.venv/bin/activate     # Linux / macOS
   ```

4. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

5. **Launch Jupyter:**
   ```bash
   python -m jupyter notebook
   ```

---

## 🔗 Related

- [Root Repository README](../README.md) — Full internship overview and progress tracker
- [Week 6: Deep Learning & Applied Project — Sprint 1](../BinX_Week_06/README.md) — Previous sprint with baseline pipeline and neural network foundations
