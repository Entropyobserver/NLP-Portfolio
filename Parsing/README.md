# Cross-Lingual Dependency Parsing

**Course:** 5LN700 | **Author:** Xiaojing Yang | **Date:** March 2025

\---

## Project Overview

This project investigates the impact of **transfer language selection** on cross-lingual dependency parsing for Swedish. Three experimental conditions are compared:

|Model|Training Data|Final LAS|
|-|-|-|
|Swedish Monolingual (baseline)|100 Swedish sentences|25.20|
|Swedish + Chinese (CTRF)|100 Swedish + 500 Chinese|35.56|
|Swedish + Norwegian (GTRF)|100 Swedish + 500 Norwegian|49.56|

**Key finding:** Typologically close languages (Norwegian) transfer far more effectively than distant ones (Chinese), even when data volume is equal.

\---

## File Structure

```
parsing-lab/
│
├── parsing\_lab.pdf              # Full experiment report (English)
├── parsing\_report\_bilingual.tex # LaTeX source — bilingual report (EN + ZH)
│
├── epoch\_comparison.png         # LAS learning curves (all 3 models, 30 epochs)
├── parse1.png                   # Error analysis — Example 1 (relative clause)
├── parse2.png                   # Error analysis — Example 2 (passive construction)
├── parse3.png                   # Error analysis — Example 3 (copular construction)
├── parse4.png                   # Error analysis — Example 4 (adverbial/possessive)
├── parse5.png                   # Error analysis — Example 5 (additional example)
│
├── las\_results1.txt             # Per-epoch LAS scores — Swedish monolingual
├── las\_results3.txt             # Per-epoch LAS scores — Swedish + Chinese
├── las\_results4.txt             # Per-epoch LAS scores — Swedish + Norwegian
│
├── 1.py                         # Script: extract LAS from results1 directory
├── 3.py                         # Script: extract LAS from results3 directory
├── 4.py                         # Script: extract LAS from results4 directory
└── plot.py                      # Script: plot learning curves (epoch\_comparison.png)
```

\---

## Experiment Setup

* **Parser:** `barchybrid` (transition-based dependency parser)
* **Data source:** Universal Dependencies v2.15

  * Path: `/common/student/corpora/ud-treebanks-v2.15`
* **Epochs:** 30 (one model checkpoint saved per epoch)
* **Best checkpoint:** `barchybrid.model28` (used for qualitative evaluation)
* **Evaluation metrics:** UAS, LAS, Weighted LAS

### Treebanks Used

|Language|Treebank|Train|Dev|
|-|-|-|-|
|Swedish (target)|UD\_Swedish-Talbanken|100|100|
|Norwegian (GTRF)|UD\_Norwegian-Bokmaal|500|100|
|Chinese (CTRF)|UD\_Chinese-GSDSimp|500|100|

\---

## How to Reproduce

### 1\. Extract LAS scores from training output

```bash
python 1.py   # Swedish monolingual → las\_results1.txt
python 3.py   # Swedish + Chinese   → las\_results3.txt
python 4.py   # Swedish + Norwegian → las\_results4.txt
```

> \*\*Note:\*\* Update the `directory` path inside each script to point to your local results folder before running.

### 2\. Plot learning curves

```bash
pip install matplotlib numpy
python plot.py
# Output: epoch\_comparison.png
```

\---

## How to Compile the LaTeX Report

The bilingual report (`parsing\_report\_bilingual.tex`) supports **pdfLaTeX**.

### On Overleaf

1. Upload `parsing\_report\_bilingual.tex` and `epoch\_comparison.png` to the same Overleaf project.
2. Keep the compiler set to **pdfLaTeX** (default).
3. Click **Compile**.

### Locally

```bash
pdflatex parsing\_report\_bilingual.tex
pdflatex parsing\_report\_bilingual.tex   # run twice for correct TOC/references
```

> \*\*Dependency:\*\* Requires the `CJKutf8` package (included in most standard TeX distributions such as TeX Live and MiKTeX).

\---

## Results Summary

### Quantitative (Epoch 30)

|Model|UAS|LAS|Weighted LAS|
|-|-|-|-|
|Swedish Monolingual|44.64|25.86|19.31|
|Swedish + Chinese|48.90|35.56|29.46|
|Swedish + Norwegian|61.06|49.56|43.16|

### Qualitative Error Patterns

|Error Type|Description|
|-|-|
|Relative clause attachment|*som*, *där* attached to wrong heads|
|Passive construction|`nsubj:pass` confused with `advmod`|
|Prepositional phrase \& copula|Prepositions mislabelled; copular verbs misidentified|

\---

## References

* Nivre, J. et al. (2016). Universal Dependencies v1. *LREC 2016*.
* Universal Dependencies Consortium (2024). UD v2.15. https://universaldependencies.org/
* Ballesteros, M. \& Nivre, J. (2013). Going to the Roots of Dependency Parsing. *Computational Linguistics*, 39(1).

