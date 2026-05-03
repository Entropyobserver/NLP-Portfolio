# Detecting Figurative Language in Cinematic Dialogue

### A Corpus Annotation and Classification Study

> \\\*\\\*ML4 Joint-Consolidated Annotation Project\\\*\\\*  
> Authors:   Xiaojing YANG ·Leyou LI · Yijie LIU

\---

## &#x20;Project Overview

This project builds a binary text classifier to distinguish **figurative language** (metaphors, idioms, and related devices) from **literal language** in movie and TV dialogue.  
Each sentence is labelled by three annotators and resolved via joint discussion into a single gold-standard label.

|Label|Meaning|Example|
|-|-|-|
|`0`|Literal language|*"Send forth all legions."*|
|`1`|Figurative language|*"Trust is earned, not given away."*|

\---

## Repository Structure

```
.
├── data/
│   ├── training.csv                  # Raw annotated training data (7 columns)
│   ├── training\\\_ne.csv               # Training subset (no-entity filtered)
│   ├── test.csv                      # Raw annotated test data
│   ├── cleaned\\\_training.csv          # Cleaned training set  \\\[Label, text]
│   └── cleaned\\\_test.csv              # Cleaned test set      \\\[Label, text]
│
├── outputs/
│   └── incorrect\\\_predictions.csv     # Misclassified samples from best SVM
│
├── annotation\\\_ml.ipynb              # Main experiment notebook
├── report\\\_anna.pdf         # Compiled PDF report
└── README.md
```

\---

## Dataset Statistics

|Split|Total|Label 0 (Literal)|Label 1 (Figurative)|
|-|-|-|-|
|Training|419|285 (68.0%)|134 (32.0%)|
|Test|98|65  (66.3%)|33  (33.7%)|
|**Total**|**517**|**350 (67.7%)**|**167 (32.3%)**|

> ⚠️ The dataset has a \\\*\\\*\\\~2:1 class imbalance\\\*\\\* (literal vs. figurative).

\---

## Methods

### 1\. Data Preprocessing

* Strip whitespace · lowercase normalisation
* Extract `text` and `Label` columns from raw 7-column annotation files

### 2\. Majority-Class Baseline

* Predict Label 0 for every sample → **66.3% accuracy, 0% recall on Label 1**

### 3\. TF-IDF + LinearSVC *(primary model)*

* Grid search over vocabulary size, n-gram range, regularisation `C`, and loss function
* 5-fold cross-validation scored by weighted F1
* Best result: **\~63% accuracy · weighted F1 \~0.62**

### 4\. Multi-Classifier Benchmark

14 classifiers tested under identical TF-IDF conditions:
`Logistic Regression` · `SVM (Linear/RBF)` · `Decision Tree` · `Random Forest` · `KNN` · `Gradient Boosting` · `XGBoost` · `LightGBM` · `CatBoost` · `AdaBoost` · `Multinomial/Bernoulli/Complement NB`

### 5\. BERT Fine-Tuning *(exploratory)*

* `bert-base-uncased` with a linear classification head
* AdamW · lr = 2e-5 · batch size 16 · 10 epochs
* Best checkpoint selected by validation weighted F1

\---

## Key Results

|Model|Accuracy|Weighted F1|Label-1 Recall|
|-|-|-|-|
|Majority Baseline|66.3%|0.53|0.00|
|LinearSVC (tuned)|\~63%|\~0.62|**0.33**|
|Logistic Regression|\~62%|\~0.61|\~0.30|
|BERT (fine-tuned)|TBD|TBD|TBD|

\---

## Error Analysis

The best SVM model produces **33 misclassified samples**:

|Error Type|Count|Share|
|-|-|-|
|False Negative (figurative → literal)|31|93.9%|
|False Positive  (literal → figurative)|2|6.1%|

**Root causes:** class imbalance · lexical similarity between classes · short utterances with minimal signal · lack of dialogue context in TF-IDF features.

\---

## Getting Started

### Requirements

```bash
pip install pandas scikit-learn numpy matplotlib seaborn
pip install torch transformers          # for BERT
pip install xgboost lightgbm catboost   # for the classifier benchmark
```

### Run the notebook

```bash
jupyter notebook annotation\\\_ml4.ipynb
```

Make sure `cleaned\\\_training.csv` and `cleaned\\\_test.csv` are in the same directory as the notebook before running.

\---

## Report

The full technical report (English + 中文) is available as:

* **LaTeX source**: `report\\\_annotation\\\_ml4.tex`
* **Compiled PDF**: `report\\\_annotation\\\_ml4.pdf`

\---

## Authors

|Name|Role|
|-|-|
|Xiaojing YANG|Annotation · Modelling · Report|
|Leyou LI|Annotation · Modelling · Report|
|Yijie LIU|Annotation · Modelling · Report|



