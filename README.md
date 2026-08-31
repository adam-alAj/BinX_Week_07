# Week 7: Deep Learning & Applied Project — Sprint 2 (Convolution & Pipeline Improvement)

**Welcome to Week 7 of the BinX Tech AI & Machine Learning Internship Program.**
This week is **Sprint 2** of **Phase 3 — Deep Learning & Applied Project**. Sprint 1 (Week 6) established a robust, reproducible ML baseline pipeline on the Heart Disease dataset. Sprint 2 focuses on understanding convolution and the architecture-selection principle, then improving the classification pipeline through feature engineering, cross-validation, and systematic hyperparameter tuning.

---

## 📋 Table of Contents

| Day | Topic | Notebook | Status |
|:---:|-------|----------|:------:|
| 1 | Sprint 2 Kickoff & Convolution | [`Sprint2-Kickoff_Convolution.ipynb`](./Day1/Sprint2-Kickoff_Convolution.ipynb) | ✅ |
| 2 | Building CNNs & Transfer Learning | [`Building_CNNs.ipynb`](./Day2/Building_CNNs.ipynb) | ✅ |
| 3 | — | — | ⬜ |
| 4 | — | — | ⬜ |
| 5 | — | — | ⬜ |

---

## 📖 Summary by Day

### [Day 1 — Sprint 2 Kickoff & Convolution](./Day1/README.md)
Completed **Sprint 2 planning** with a 7-task backlog and acceptance criteria, then learned convolution through a hands-on edge-detection demonstration. Reviewed the Sprint 1 retrospective and defined the Sprint 2 goal: improve the heart-disease classification pipeline through feature engineering, cross-validation, and systematic tuning. Compared the three main deep learning architectures (CNN for images, RNN/Transformer for sequences, Dense for tabular) and confirmed that the Heart Disease dataset (918 patients × 14 columns) is **tabular** — making the **dense (fully connected) network** the correct choice. Created a 256×256 synthetic grayscale image with geometric shapes and applied two hand-crafted 3×3 edge-detection kernels (horizontal and vertical) via `scipy.ndimage.convolve`. Visualized the original image, both feature maps, and the combined gradient magnitude in a 2×3 subplot grid. Demonstrated numerically why convolution is parameter-efficient: a 3×3 kernel with **9 parameters** produces a feature map across 65,025 output positions, while a dense layer would require **4.2 billion parameters** — a ~470 million× difference from parameter sharing. The notebook also defines the Sprint 2 backlog (7 tasks with priorities and acceptance criteria), core-model targets (beat Accuracy > 0.8533, F1 > 0.8657, ROC-AUC > 0.9159), and the full sprint events timeline.

### [Day 2 — Building CNNs & Transfer Learning](./Day2/README.md)
Completed the required computer-vision educational lab using a **skin lesion image dataset** (11,879 train + 2,000 test images, binary Benign vs Malignant classification). Built and compared three approaches: (1) **CNN From Scratch** — a 3-block Conv2D→MaxPool network trained on 128×128 images with ~4.3M parameters; (2) **CNN + Data Augmentation** — same architecture with RandomFlip, RandomRotation, and RandomZoom applied during training via the Functional API; (3) **Transfer Learning** — frozen MobileNetV2 backbone (ImageNet pre-trained) with a new GlobalAveragePooling2D→Dense classification head trained on 224×224 images. All three experiments use identical train/val/test splits, Adam optimizer, binary cross-entropy loss, and EarlyStopping. Includes training curves, baseline-vs-augmentation comparison, three-way experiment table, confusion matrix, and classification report. Clearly documented that this CNN lab is an educational exercise — the project's core model remains the **dense network from Week 6** for the tabular Heart Disease dataset. Dataset: [Melanoma Skin Cancer — Benign vs Malignant](https://www.kaggle.com/datasets/ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant) (Kaggle).

### Day 3 — *(Pending)*

### Day 4 — *(Pending)*

### Day 5 — *(Pending)*

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
├── Day4/
├── Day5/
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
