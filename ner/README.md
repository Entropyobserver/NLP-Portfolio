# Named Entity Recognition and Classification (NERC)

A machine learning pipeline for identifying and classifying named entities in text, built on the **CoNLL-2003** benchmark corpus. The project progresses from a simple feature-based classifier to an ensemble model and a neural BiLSTM-CRF architecture.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Overview

Named Entity Recognition (NER) is the task of locating and classifying named entities in raw text into predefined categories such as person names, organisations, locations, and miscellaneous proper nouns. This lab explores how feature engineering, model selection, and post-processing affect NER performance on a realistic, class-imbalanced dataset.

Entity tags follow the **IOB2** scheme:
- `B-` prefix — first token of an entity
- `I-` prefix — continuation token of the same entity
- `O` — not part of any named entity

---

## Dataset

**CoNLL-2003** — Reuters newswire corpus annotated with four entity types.

| Split      | Sentences |
|------------|----------:|
| Train      | 14,041    |
| Validation | 3,250     |
| Test       | 3,453     |
| **Total**  | **20,744**|

Downloaded programmatically via Hugging Face `datasets`:

```python
from datasets import load_dataset
dataset = load_dataset("conll2003", trust_remote_code=True)
```

**NER tag set (9 classes):** `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`

**Sentence length** (training split): min 1 · max 113 · mean 14.5 · median 10 tokens

> **Note:** The dataset is heavily imbalanced — `O` tokens vastly outnumber entity tokens. F1-score is used as the primary evaluation metric rather than accuracy.

---

## Project Structure

```
NERC/
├── NERC.ipynb               # Main notebook (all experiments)
├── lab1_NERC.pdf            # Lab report with results and analysis
├── README.md
└── requirements.txt
```

---

## Features

### Base Features (per token)
| Feature | Description |
|---------|-------------|
| `token` | The word itself — provides lexical identity |
| `is_capitalized` | Whether the first letter is uppercase — strong signal for proper nouns |
| `pos_tag` | Part-of-speech tag — encodes grammatical role |

### Extended Features (VG section)
| Feature | Description |
|---------|-------------|
| Word embeddings | Pre-trained GloVe vectors; zero vectors as fallback |
| Token length | Number of characters |
| Pattern flags | Boolean flags for dates, times, monetary values, e-mail addresses |
| Gazetteer membership | Whether the token appears in an entity dictionary built from training data |

### Additional options discussed
- Character-level features (prefixes, suffixes) — handles out-of-vocabulary words
- Word shape (e.g. `Xxxx`, `ALL-CAPS`) — useful for acronyms and names
- BERT / contextual embeddings — deep semantic context

> **Data leakage rule:** `fit()` and `fit_transform()` are called **only on training data**. `transform()` is used on validation and test sets.

---

## Models

### 1. LinearSVC (Baseline)
A linear support vector classifier trained on the base feature set.

```python
from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 2. Extended Model — Entity Dictionaries + GloVe
- `create_entity_dictionaries()` extracts known entity strings from the training set and provides a lookup signal at inference time.
- GloVe vectors are concatenated to the feature vector, adding semantic context beyond surface form.

### 3. Ensemble Model — SVM + Logistic Regression + Random Forest
Three classifiers trained independently and combined via **majority voting**. Hyperparameter tuning (e.g. number of trees, tree depth) and k-fold cross-validation are applied.

### 4. BiLSTM-CRF (Neural)
A PyTorch sequence labelling model:

```
Token IDs → Embedding (100-d) → BiLSTM (2 layers, hidden 256) → Linear → CRF → Tags
```

- **BiLSTM** captures left and right context simultaneously.
- **CRF layer** enforces tag-transition constraints (e.g. `I-PER` cannot follow `B-LOC`), ensuring globally consistent label sequences. Training uses the forward algorithm (negative log-likelihood); decoding uses Viterbi search.
- A **BERT-CRF** variant (`BertCRFForNER`) fine-tuning `bert-base-cased` is also implemented.

---

## Results

### Baseline LinearSVC

| Class  | Precision | Recall | F1     |
|--------|----------:|-------:|-------:|
| O      | 0.9924    | 0.9803 | 0.9863 |
| B-PER  | 0.3667    | 0.8899 | 0.5194 |
| I-PER  | 0.5876    | 0.1886 | 0.2855 |
| B-ORG  | 0.7956    | 0.5226 | 0.6308 |
| I-ORG  | 0.6890    | 0.4563 | 0.5490 |
| B-LOC  | 0.8080    | 0.7770 | 0.7922 |
| I-LOC  | 0.6199    | 0.5331 | 0.5732 |
| B-MISC | 0.7809    | 0.6652 | 0.7185 |
| I-MISC | 0.5845    | 0.5926 | 0.5885 |
| **Mean** | **0.6916** | **0.6228** | **0.6270** |

Overall NER F1 (all entity tags): **0.6757**

### Extended Model (Dictionaries + GloVe)

Overall F1 improves from 0.6044 → **0.6270**, driven by a +5.4% recall gain. B-PER precision rises substantially (0.37 → 0.75), though I-ORG and I-LOC precision drops.

### Ensemble (SVM + LR + RF)

Overall test F1: **0.6313** · Cross-validation F1: **0.9277 ± 0.0039**

> ⚠️ The large gap between cross-validation and test F1 indicates significant **overfitting**. The ensemble achieves high precision but very low recall for PER entities (F1 ≈ 0.31).

### Key Findings

| Entity | Difficulty | Reason |
|--------|------------|--------|
| `O` | Easy | Majority class; no entity structure required |
| `LOC` | Moderate | Distinctive capitalisation and geographic conventions |
| `ORG` | Hard | Structurally varied; overlaps with common nouns (e.g. "Apple") |
| `PER` | Hardest | Diverse names; overlap with common words; multi-token names difficult |
| `I-*` tags | Harder than `B-*` | Errors propagate from the boundary token |

---

## Requirements

```
datasets>=2.0
scikit-learn>=1.0
torch>=1.12
transformers>=4.20
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

Open and run `NERC.ipynb` section by section:

1. **Section 1** — Download and explore the CoNLL-2003 dataset
2. **Section 2** — Build tag dictionaries and inspect samples
3. **Section 3** — Extract features with `extract_data()`
4. **Section 4** — Train and evaluate the baseline LinearSVC
5. **Section 5 (VG)** — Extended features, ensemble model, and BiLSTM-CRF

All models follow the same interface:

```python
# Train
model.fit(X_train, y_train)

# Evaluate — always transform, never fit, on test data
X_test_transformed = vectorizer.transform(X_test)
predictions = model.predict(X_test_transformed)
```
