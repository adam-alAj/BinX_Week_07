# Day 2 — Building CNNs & Transfer Learning

Welcome to Day 2 of Week 7. This session continues **Sprint 2** within **Phase 3 — Deep Learning & Applied Project**. Building on Day 1's introduction to convolution and parameter sharing, we now construct a **complete trainable CNN from scratch**, apply **data augmentation** to reduce overfitting, and use **transfer learning** with a pre-trained MobileNetV2 to leverage features learned from ImageNet.

---

## 🎯 Objective

- Build a full CNN with convolution, pooling, and dense layers from scratch and record its accuracy
- Apply data augmentation (RandomFlip, RandomRotation, RandomZoom) and compare validation curves against the baseline
- Use transfer learning with a frozen pre-trained MobileNetV2 and compare accuracy and training time
- Document which approach performed best and explain why using actual experimental evidence

---

## 📓 Notebook

- [Building_CNNs.ipynb](./Building_CNNs.ipynb)

---

## 🗂️ Dataset

| Property | Value |
|:---------|:------|
| **Task** | Binary classification: Benign vs Malignant (skin lesion images) |
| **Training samples** | 11,879 (6,289 Benign + 5,590 Malignant) |
| **Test samples** | 2,000 (1,000 Benign + 1,000 Malignant) |
| **Image size** | 224 × 224 pixels, 3 channels (RGB JPEG) |
| **Train/Val split** | 80/20 (≈9,503 train / ≈2,376 val) |

> **Why this dataset?** The project's Heart Disease dataset is **tabular** (918 patients × 14 columns), so CNNs are not the correct architecture for the core model. This Day 2 lab uses **real medical images** (skin lesion classification) — a task where CNNs are the correct architecture. The medical imaging context is relevant to the broader healthcare domain of the Cardiac Patient Monitoring System project.

### 📥 Download Instructions

1. Go to the Kaggle dataset page:
   [https://www.kaggle.com/datasets/ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant](https://www.kaggle.com/datasets/ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant)

2. Click **Download** (you need a free Kaggle account).

3. Extract the downloaded zip so that the folder structure matches:
   ```
   Data/images-dataset/
   ├── train/
   │   ├── Benign/
   │   │   ├── 1.jpg
   │   │   └── ...
   │   └── Malignant/
   │       ├── 1.jpg
   │       └── ...
   └── test/
       ├── Benign/
       │   └── ...
       └── Malignant/
           └── ...
   ```

> **Note:** The dataset must be placed at `Data/images-dataset/` relative to the repository root for the notebook paths to work.

---

## 🏗️ Experiments

### Experiment 1 — CNN From Scratch

A complete CNN built from scratch with three convolutional blocks:

```
Input (128×128×3)
↓
Conv2D (32 filters, 3×3, ReLU) + MaxPooling2D (2×2)
↓
Conv2D (64 filters, 3×3, ReLU) + MaxPooling2D (2×2)
↓
Conv2D (128 filters, 3×3, ReLU) + MaxPooling2D (2×2)
↓
Flatten → Dense (128, ReLU) → Dropout (0.5) → Dense (1, Sigmoid)
```

- Images resized to 128×128 for faster CPU training
- Optimizer: Adam | Loss: Binary Cross-Entropy | Epochs: up to 15 (EarlyStopping, patience=5)
- **Total parameters:** ~4,288,000

### Experiment 2 — CNN + Data Augmentation

Same architecture as Experiment 1, but with an augmentation pipeline applied **only during training**:

| Transform | Rate | Purpose |
|:----------|:-----|:--------|
| **RandomFlip** | horizontal | Left-right mirroring invariance |
| **RandomRotation** | ±15% | Slight tilt invariance |
| **RandomZoom** | ±15% | Scale invariance |

Built using the **Functional API** because the augmentation `Sequential` model is nested inside the main model — the Functional API correctly traces shapes through nested layers.

### Experiment 3 — Transfer Learning (MobileNetV2)

A pre-trained MobileNetV2 backbone with a new classification head:

```
MobileNetV2 (frozen — ImageNet weights, weights NOT updated)
↓
GlobalAveragePooling2D
↓
Dense (128, ReLU) → Dropout (0.3) → Dense (1, Sigmoid)
```

- **Input size:** 224×224×3 (matches raw image size)
- **Preprocessing:** MobileNetV2 expects pixels in [-1, 1]
- **Frozen backbone:** All pre-trained weights are frozen — only the new head is trained
- **Optimizer:** Adam (lr=0.001) | Epochs: up to 10 (EarlyStopping, patience=5)

---

## 📊 Experiment Comparison

| Model | Augmentation | Transfer Learning | Backbone | Best Val Acc | Test Acc | Time |
|:------|:------------|:-----------------|:---------|:-------------|:---------|:-----|
| CNN From Scratch | No | No | N/A | See notebook | See notebook | See notebook |
| CNN + Augmentation | Yes | No | N/A | See notebook | See notebook | See notebook |
| Transfer Learning | Appropriate | Yes | Frozen | See notebook | See notebook | See notebook |

> **Note:** Exact metrics are populated when the notebook is executed. Run the notebook to see actual results.

---

## 📈 Visualizations

The notebook produces the following plots:

1. **Sample images** from the dataset (Benign vs Malignant)
2. **Data augmentation examples** — original + 9 augmented versions
3. **Training curves** for each experiment (accuracy + loss vs epoch)
4. **Baseline vs Augmentation** side-by-side validation curves
5. **All three experiments** validation curves comparison
6. **Bar charts** comparing accuracy and training time across experiments
7. **Confusion matrix** for the best-performing model
8. **Classification report** (precision, recall, F1-score) for the best model

---

## ✅ Key Tasks & Accomplishments

- **CNN From Scratch:** Built a complete 3-block CNN (Conv2D → MaxPool × 3 → Flatten → Dense) trained on 128×128 skin lesion images. The model successfully learned visual features (edges, textures, patterns) without any external pre-training.

- **Data Augmentation:** Implemented a `keras.Sequential` augmentation pipeline (RandomFlip, RandomRotation, RandomZoom) and integrated it using the Functional API to avoid the nested-Sequential shape-inference issue. Augmentation is applied only during training — validation and test sets remain unaugmented.

- **Transfer Learning:** Used MobileNetV2 pre-trained on ImageNet as a frozen feature extractor. Only the new classification head (GlobalAveragePooling2D → Dense → Dropout → Dense) is trained. Input images are preprocessed to [-1, 1] as MobileNetV2 expects.

- **Fair Comparison:** All three experiments use the same train/val/test splits, the same binary cross-entropy loss, and the same Adam optimizer. Training time is measured for each approach.

- **Architecture Decision:** Clearly documented that the CNN work is an **educational computer-vision lab** using real medical images. The project's core model remains the **dense network from Week 6** for the tabular Heart Disease dataset. CNNs are NOT forced onto tabular data.

---

## 🛠️ Skills Covered

- **Conv2D layers:** Learned filters for feature extraction (edges → textures → patterns)
- **MaxPooling2D:** Spatial downsampling, translation invariance, computation reduction
- **Flatten + Dense:** Converting 2D feature maps to 1D classification head
- **Data Augmentation:** RandomFlip, RandomRotation, RandomZoom as regularization
- **Transfer Learning:** MobileNetV2 with frozen backbone, new classification head
- **Functional API:** Composing nested Sequential models correctly
- **Image Data Loading:** `tf.keras.utils.image_dataset_from_directory`
- **MobileNetV2 Preprocessing:** Pixel normalization to [-1, 1]
- **EarlyStopping:** Preventing wasteful training epochs
- **Model Evaluation:** Accuracy, loss, confusion matrix, classification report
- **Training Curves:** Diagnosing overfitting from train-validation gaps
- **TensorFlow/Keras:** Sequential, Functional API, callbacks, model.compile/fit/evaluate

---

## 🔗 Related

- [Day 1 — Sprint 2 Kickoff & Convolution](../Day1/README.md)
- [Week 7 Overview](../README.md)
- [Root Repository README](../../README.md)
