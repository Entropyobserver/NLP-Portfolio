# RNN-based Part-of-Speech Tagger

A PyTorch implementation of a Part-of-Speech (POS) tagger built on the **Universal Dependencies (UD)** corpus. The project starts from a plain LSTM baseline and progressively adds bidirectionality, better data pipelines, augmentation, regularisation, and multilingual support — reaching **≥ 98% accuracy** on English.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Experiments](#experiments)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Overview

Part-of-Speech tagging assigns a grammatical category (noun, verb, adjective, etc.) to every token in a sentence. It is a foundational NLP task used in parsing, information extraction, and downstream language understanding.

This project implements a sequence labelling pipeline:

```
Input tokens → Embedding → RNN/LSTM/GRU → Linear → Log-Softmax → POS tags
```

Tags follow the **Universal POS (UPOS)** tagset — 17 coarse-grained categories shared across all languages — as well as language-specific **XPOS** tags in later experiments.

---

## Dataset

**Universal Dependencies (UD)** — morphosyntactically annotated corpora in a consistent, cross-linguistically applicable framework.

Primary dataset: **English Web Treebank** (`en_ewt`).  
Later experiments extend to six languages:

| Language | Dataset     |
|----------|-------------|
| English  | `en_ewt`    |
| French   | `fr_gsd`    |
| German   | `de_gsd`    |
| Japanese | `ja_gsd`    |
| Finnish  | `fi_tdt`    |
| Chinese  | `zh_gsd`    |

Data is split **90% train / 10% test** with a fixed random seed. Sentences with fewer than 3 tokens are filtered out. Vocabulary includes `<UNK>` (for unknown words) and `<PAD>` (for batch padding) special tokens.

---

## Project Structure

```
postagging/
├── postagging.ipynb         # Main notebook (all experiments)
├── README.md
└── requirements.txt
```

---

## Model Architecture

### Core Model — `RNNTagger`

A flexible tagger supporting three recurrent cell types and optional bidirectionality:

```
Token indices
    │
    ▼
Embedding layer  (dim = d_e)
    │
    ▼
RNN / LSTM / GRU  (hidden = d_h, bidirectional optional)
    │
    ▼
Dropout + BatchNorm
    │
    ▼
Linear  (d_h × num_directions → num_tags)
    │
    ▼
Log-Softmax → tag probabilities
```

| Component | Detail |
|-----------|--------|
| Embedding | Trainable, Xavier-initialised |
| Recurrent | RNN / LSTM / GRU, configurable layers |
| Bidirectional | Doubles hidden dim fed to linear layer |
| Dropout | Applied after embedding and before output |
| Batch Norm | Applied to flattened RNN outputs |
| Loss | NLL loss with padding ignored (`ignore_index=pad_idx`) |
| Optimiser | Adam with optional weight decay |

Sequences are packed with `pack_padded_sequence` / `pad_packed_sequence` to avoid computing states over padding tokens.

---

## Experiments

### Baselines

| Baseline | Strategy |
|----------|----------|
| Most-Common-Tag | Assign the globally most frequent tag to every token |
| Dictionary | Assign per-word most frequent tag from training set; fall back for unseen words |

### Section 6.1 — Architecture Comparison

Vanilla RNN, LSTM, and GRU evaluated under identical hyperparameters  
(`d_e = 32`, `d_h = 64`, batch 256, lr 0.01, 5 epochs).

### Section 6.2 — Bidirectional RNN

`bidirectional=True` concatenates forward and backward hidden states.  
Linear layer input adjusted: `nn.Linear(hidden_dim * 2, tagset_size)`.

### Section 6.3 — PyTorch DataLoader

Manual batch iterator replaced with `POSDataset` + `DataLoader`.  
`collate_fn` pads each batch dynamically to its own maximum length.  
**Result: accuracy jumps from ~89% to ~97%.**

### Section 6.4 — Data Augmentation

Token masking: each token replaced by `<UNK>` with probability `p = 0.15` during training, forcing the model to rely on context rather than memorised word–tag pairs.

### Section 6.5 — Regularisation

| Technique | Detail |
|-----------|--------|
| Dropout | Rate 0.3, after embedding and before output |
| L2 weight decay | `weight_decay=1e-4` in Adam |
| Batch normalisation | On flattened LSTM outputs |
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` |
| LR scheduling | `ReduceLROnPlateau` — halve lr on val-loss plateau |
| Early stopping | Patience = 3 epochs; best checkpoint restored |
| Weight init | Xavier / Glorot for linear and embedding layers |

### Section 6.6 — Multi-Source Training

Four configurations tested: English only → multi-English → multi-language (EN + FI + DE) → all sources (10 languages). Adding English data helps; cross-lingual mixing slightly reduces English accuracy but improves cross-lingual robustness.

### Section 6.7 — Multilingual Comparison

The fully optimised model trained and evaluated separately on six languages.  
UPOS vs. XPOS comparison: UPOS wins for analytic languages (EN, FR, DE);  
XPOS wins for morphologically rich languages (FI, JA).

---

## Results

### Architecture Comparison (`en_ewt`)

| Architecture | Train Acc. | Test Acc. |
|--------------|:----------:|:---------:|
| Vanilla RNN  | 95.8%      | 88.6%     |
| LSTM         | 98.6%      | 89.2%     |
| GRU          | 98.4%      | 89.2%     |

### Progressive Improvements (`en_ewt`)

| Configuration | Test Acc. |
|---------------|:---------:|
| Most-Common-Tag baseline | ~20% |
| Dictionary baseline | ~85% |
| LSTM (manual batching) | 89.2% |
| BiLSTM (manual batching) | ~89.5% |
| BiLSTM + DataLoader | ~97% |
| BiLSTM + DataLoader + augmentation | 97.9–98.0% |
| BiLSTM + DataLoader + augmentation + regularisation | ≥ 98% |

### Multilingual Results (optimised model, 5 epochs)

| Language | Dataset   | Test Acc.  |
|----------|-----------|:----------:|
| English  | `en_ewt`  | ~98.0%     |
| French   | `fr_gsd`  | ~98.0%     |
| German   | `de_gsd`  | ~97.0%     |
| Japanese | `ja_gsd`  | ~98.0%     |
| Finnish  | `fi_tdt`  | ~97.7%     |
| Chinese  | `zh_gsd`  | ~95.2%     |

### UPOS vs. XPOS

| Language | Winner | Margin |
|----------|--------|-------:|
| English  | UPOS   | +0.5%  |
| German   | UPOS   | +0.6%  |
| Chinese  | Tied   | —      |
| Japanese | XPOS   | —      |
| Finnish  | XPOS   | +1.6%  |

---

## Requirements

```
torch>=1.12
datasets>=2.0
numpy
matplotlib
seaborn
tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Usage

Open and run `postagging.ipynb` section by section:

1. **Section 1–5** — Data loading, vocabulary construction, baseline models, and core LSTM tagger
2. **Section 6.1** — Swap recurrent cell: set `rnn_type='RNN'|'LSTM'|'GRU'`
3. **Section 6.2** — Enable bidirectionality: `bidirectional=True`
4. **Section 6.3** — Switch to DataLoader pipeline
5. **Section 6.4** — Enable token masking: `mask_prob=0.15`
6. **Section 6.5** — Add full regularisation suite
7. **Section 6.6–6.7** — Multi-source and multilingual experiments

Quick training example:

```python
model = RNNTagger(
    vocab_size=len(word2idx),
    tagset_size=len(tag2idx),
    embedding_dim=128,
    hidden_dim=256,
    rnn_type='LSTM',
    bidirectional=True,
    dropout=0.3,
)

train(model, train_loader, val_loader,
      epochs=10, lr=1e-3, weight_decay=1e-4,
      clip_norm=1.0, patience=3)
```
