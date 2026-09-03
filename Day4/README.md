# Day 4 — Attention & Transformers

Welcome to Day 4 of Week 7. This session continues **Sprint 2** within **Phase 3 — Deep Learning & Applied Project**. Completing the Week 7 architectural quartet (Convolution → CNNs → RNNs/LSTMs → Transformers), we shift from sequential signal processing (Day 3) to **natural language processing** — fine-tuning a pre-trained AraBERT Transformer on Arabic sentiment reviews and making the final architecture decision for the project.

---

## 🎯 Objective

1. Explain why RNNs/LSTMs have fundamental limitations that attention mechanisms solve
2. Describe the Transformer architecture (self-attention, multi-head attention, positional encoding)
3. Explain why pre-trained Transformers (BERT) work well for text classification
4. Load and configure a pre-trained AraBERT v2 model for Arabic sentiment classification
5. Fine-tune AraBERT on the 330K Arabic Sentiment Reviews dataset using the Hugging Face Trainer API
6. Evaluate the Transformer and compare its architectural properties against the Day 3 LSTM
7. Make an evidence-based architecture decision for the project's core model

---

## 📓 Notebooks

- [Attention_Transformers.ipynb](./Attention_Transformers.ipynb) — Colab GPU Edition (recommended)
- [Attention_Transformers1.ipynb](./Attention_Transformers1.ipynb) — Alternative version

---

## 🗂️ Dataset

| Property | Value |
|:---------|:------|
| **Task** | Binary sentiment classification (positive / negative) |
| **Source** | [330K Arabic Sentiment Reviews](https://www.kaggle.com/datasets/abdallaellaithy/330k-arabic-sentiment-reviews) (Kaggle) |
| **Total samples** | 330,000 Arabic product reviews |
| **Sampled for training** | 20,000 (stratified, 50/50 class balance) |
| **Train / Val / Test** | 14,000 / 3,000 / 3,000 |
| **Language** | Arabic (morphologically rich, dialectal variations) |
| **Class balance** | Nearly balanced — 49.4% Negative, 50.6% Positive |

> **Why this dataset?** Arabic sentiment classification is a natural domain for Transformers because: (1) Arabic is morphologically rich where contextual understanding matters; (2) negation scope and context-dependent sentiment require long-range attention; (3) pre-trained Arabic BERT models provide deep linguistic knowledge from massive corpora.

---

## 🏗️ Architecture Overview

### Pre-trained Model: AraBERT v2 (`aubmindlab/bert-base-arabertv2`)

| Property | Value |
|:---------|:------|
| **Architecture** | BERT-base (12 layers, 768 hidden, 12 attention heads) |
| **Parameters** | ~110M |
| **Pre-training data** | 67M Arabic sentences from Arabic Open Web (Common Crawl) |
| **Tokenizer** | WordPiece with 64K vocabulary |
| **Max sequence length** | 512 tokens (128 used for this experiment) |

### Why AraBERT v2?

1. **Arabic-specific** — Pre-trained exclusively on Arabic text, capturing morphology, diacritics, and dialectal variations
2. **Pre-trained knowledge** — 67M Arabic sentences provide deep linguistic understanding "for free"
3. **Self-attention** — Handles long-range dependencies (negation scope, context-dependent sentiment) natively
4. **Proven performance** — State-of-the-art on Arabic NLP benchmarks
5. **Practical size** — BERT-base (~110M params) is manageable for fine-tuning

---

## 🏗️ Experiments

### Experiment 1 — Transformer Fine-Tuning (AraBERT v2)

Fine-tune the pre-trained AraBERT model on the Arabic Sentiment dataset using the Hugging Face `Trainer` API:

```
Input: Arabic review text
↓
AraBERT Tokenizer (WordPiece, max_length=128)
↓
AraBERT v2 Encoder (12 layers, 12 attention heads, 768 hidden)
↓
[CLS] token representation
↓
Classification Head (Linear → 2 logits)
↓
Softmax → Negative / Positive
```

- **Training framework:** Hugging Face `Trainer` with `EarlyStoppingTrainer` subclass
- **Optimizer:** AdamW (lr=2e-5, weight_decay=0.01)
- **Scheduler:** Linear warmup (10% of steps) + linear decay
- **Batch size:** 32 (GPU) / 16 (CPU)
- **Epochs:** 3 (with early stopping, patience=1 epoch)
- **Mixed precision:** FP16 when CUDA available
- **Checkpointing:** Saved per epoch to Google Drive, keep last 2 checkpoints

### Comparison with Day 3 LSTM

The Day 3 LSTM and Day 4 Transformer were trained on **different datasets and tasks** (ECG signals vs. Arabic text), so this is **not a controlled A/B comparison**. Instead, we compare architectural properties and practical considerations.

---

## 📊 Results

### Transformer Evaluation (Test Set)

| Metric | Score |
|:-------|:------|
| **Accuracy** | *(from notebook execution)* |
| **Precision (macro)** | *(from notebook execution)* |
| **Recall (macro)** | *(from notebook execution)* |
| **F1-Score (macro)** | *(from notebook execution)* |

> Results are populated from the actual evaluation — see Section 11 of the notebook for exact values.

### Architectural Comparison: LSTM vs Transformer

| Aspect | LSTM (Day 3) | Transformer (Day 4) |
|:-------|:-------------|:---------------------|
| **Parallelization** | Sequential (1/5) | Fully parallel (5/5) |
| **Long-range dependencies** | Degrade over distance (2/5) | Direct connection (5/5) |
| **Pre-trained knowledge** | None (1/5) | 67M Arabic sentences (5/5) |
| **Training data efficiency** | Better with small data (3/5) | Needs fine-tuning data (2/5) |
| **Interpretability** | Gates are interpretable (4/5) | Attention weights less intuitive (2/5) |
| **Inference speed** | Faster for single samples (4/5) | Slower, larger model (2/5) |

### Final Architecture Decision

Based on the evidence gathered across Week 7:

> **★ RECOMMENDATION: Use the Transformer (AraBERT) as the project's core architecture for Arabic text sentiment classification.**

**Rationale:**
1. Pre-trained Arabic linguistic knowledge (67M sentences)
2. Superior handling of long-range dependencies via self-attention
3. Better morphological understanding of Arabic
4. Strong performance with limited fine-tuning data
5. Parallel processing capability for efficient training

This decision follows the **Week 7 principle: match the architecture to the data type.**
- Text → Transformer
- Signals → LSTM
- Images → CNN
- Tabular → Dense network

---

## 📈 Visualizations

1. **Label distribution** — Bar chart showing near-balanced class distribution (49.4% / 50.6%)
2. **Text length distribution** — Histogram of review lengths for sequence-length planning
3. **Tokenization demonstration** — Arabic text → WordPiece tokens → Token IDs
4. **Sample inference (before fine-tuning)** — Model predictions on raw Arabic reviews before training
5. **Training curves** — Train/Val loss, Validation accuracy, and Validation F1 vs. epoch
6. **Confusion matrix** — Count and normalized confusion matrices on test set
7. **Sample inference (after fine-tuning)** — Improved predictions on the same Arabic reviews
8. **Architectural comparison** — Bar chart comparing LSTM vs Transformer across 6 dimensions

---

## ✅ Key Tasks & Accomplishments

- **Attention vs RNN Theory:** Explained the fundamental limitations of RNN/LSTM memory (sequential bottleneck, vanishing gradients, fixed-length hidden state) and how self-attention solves them (parallel processing, direct long-range connections, dynamic relevance).

- **Transformer Architecture:** Described the full Transformer architecture — token embeddings, positional encoding, multi-head self-attention, feed-forward networks, layer normalization, residual connections, and the encoder stack — connecting each component to its purpose.

- **Pre-trained Model Loading:** Loaded AraBERT v2 (`aubmindlab/bert-base-arabertv2`) with 135M parameters via Hugging Face `AutoModelForSequenceClassification`, including tokenizer setup and GPU diagnostics.

- **Data Preparation:** Loaded the 330K Arabic Sentiment Reviews dataset, removed duplicates, and created a stratified 20,000-sample subset with 50/50 class balance. Split into Train (14,000) / Validation (3,000) / Test (3,000).

- **Tokenization:** Demonstrated the AraBERT WordPiece tokenization pipeline — Arabic text normalization, subword tokenization, special token insertion ([CLS], [SEP], [PAD]), and padding/truncation to max_length=128.

- **Sample Inference (Before Fine-Tuning):** Showed that the pre-trained model's predictions on Arabic sentiment reviews are near-random (~50-57% confidence) before task-specific fine-tuning.

- **GPU-Optimised Fine-Tuning:** Fine-tuned AraBERT using the Hugging Face `Trainer` API with FP16 mixed precision, linear warmup+decay scheduling, AdamW optimizer (lr=2e-5), and a custom `EarlyStoppingTrainer` subclass with patience-based stopping on validation F1.

- **Evaluation:** Evaluated the fine-tuned Transformer on the held-out test set with accuracy, precision (macro), recall (macro), F1-score (macro), confusion matrix, and per-class classification report.

- **Comparison with Day 3 LSTM:** Compared the Transformer against the Day 3 LSTM across architectural properties (parallelization, long-range dependencies, pre-trained knowledge, data efficiency, interpretability, inference speed) — noting this is an architectural comparison, not a controlled A/B test on the same dataset.

- **Architecture Decision:** Made an evidence-based recommendation to use AraBERT as the project's core architecture for Arabic text classification, following the Week 7 principle of matching architecture to data type.

- **Colab GPU Refactoring:** Refactored the notebook from a CPU-only manual-loop training script into a GPU-accelerated Colab-ready pipeline with automatic checkpoint persistence to Google Drive, FP16 mixed precision, and Hugging Face Trainer API.

---

## 🛠️ Skills Covered

- **Attention Mechanism:** Query-Key-Value vectors, attention scores, softmax(QK^T / √d_k)V
- **Self-Attention:** Every token attending to every other token in the sequence
- **Multi-Head Attention:** Multiple parallel attention heads capturing different relationship types
- **Transformer Architecture:** Embeddings, positional encoding, feed-forward layers, layer normalization, residual connections, encoder stack
- **Pre-trained Transformers (BERT):** Masked Language Modeling, Next Sentence Prediction, fine-tuning for downstream tasks
- **AraBERT:** Arabic-specific BERT variant pre-trained on 67M Arabic sentences
- **Hugging Face Ecosystem:** `AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`
- **WordPiece Tokenization:** Subword tokenization for morphologically rich languages
- **FP16 Mixed Precision:** GPU-accelerated training with reduced memory footprint
- **Early Stopping (Custom):** `EarlyStoppingTrainer` subclass with patience-based F1 monitoring
- **Google Drive Checkpointing:** Persistent model artifacts across Colab sessions
- **Architecture Selection:** Matching model architecture to data type (Text → Transformer, Signals → LSTM, Images → CNN)
- **Evidence-Based Decision Making:** Systematic comparison framework for architecture choices

---

## 🔗 Related

- [Day 1 — Sprint 2 Kickoff & Convolution](../Day1/README.md)
- [Day 2 — Building CNNs & Transfer Learning](../Day2/README.md)
- [Day 3 — RNNs & LSTMs for Sequential Data](../Day3/README.md)
- [Week 7 Overview](../README.md)
- [Root Repository README](../../README.md)
