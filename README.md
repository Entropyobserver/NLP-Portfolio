# Uppsala University — NLP Coursework

> \*\*🚧 This repository is continuously updated.\*\*

A personal collection of NLP lab assignments and projects completed at **Uppsala University**.
Topics span text classification, sequence labelling, bias probing, dependency parsing, and more.

\---

## Projects

> Sorted by research relevance. New entries will appear here as they are completed.

### 1\. Geographic Bias in Pretrained NLP Models

**Sentiment \& perplexity probing across 249 countries**

Probes DistilBERT (SST-2) and GPT-2 for implicit geographic bias using fixed sentence templates — holding sentence structure constant and varying only the country name. A 22× range in GPT-2 perplexity across structurally identical sentences reveals severe training data imbalances favouring high-income, English-speaking nations.

* **Models:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`), GPT-2
* **Scope:** 249 countries (ISO 3166-1), UNSD M49 regional groupings
* **Key result:** 63.5% positive / 36.5% negative sentiment split on a semantically neutral template; perplexity range 10.45 → 233.56

\---

### 2\. RNN-based Part-of-Speech Tagger

**BiLSTM sequence labeller on Universal Dependencies**

A PyTorch POS tagger built from a plain LSTM baseline up through bidirectionality, DataLoader pipelines, token masking augmentation, full regularisation, and multilingual evaluation — reaching ≥98% accuracy on English.

* **Architecture:** BiLSTM (2 layers, hidden 256) → Dropout + BatchNorm → Linear → Log-Softmax
* **Languages:** English, French, German, Japanese, Finnish, Chinese (`en\_ewt`, `fr\_gsd`, `de\_gsd`, `ja\_gsd`, `fi\_tdt`, `zh\_gsd`)
* **Key result:** Manual batching \~89% → DataLoader + augmentation + regularisation **≥98%** on `en\_ewt`

\---

### 3\. Named Entity Recognition and Classification (NERC)

**From LinearSVC to BiLSTM-CRF on CoNLL-2003**

A full NER pipeline on the CoNLL-2003 Reuters corpus (IOB2 scheme), progressing from feature-based SVM to an ensemble model and a neural BiLSTM-CRF. A BERT-CRF variant (`bert-base-cased`) is also implemented.

* **Dataset:** CoNLL-2003 — 14,041 train / 3,250 dev / 3,453 test sentences
* **Models:** LinearSVC · Ensemble (SVM + LR + RF) · BiLSTM-CRF · BERT-CRF
* **Key result:** Baseline F1 0.6757 → ensemble F1 0.6313 (high CV score, overfitting noted); BERT-CRF ongoing

\---

### 4\. Cross-Lingual Dependency Parsing

**Transfer language selection for Swedish**

Compares three training conditions for a transition-based dependency parser (`barchybrid`) on Swedish: monolingual, Swedish + Chinese (culturally/typologically remote), and Swedish + Norwegian (typologically close).

* **Parser:** barchybrid · **Data:** Universal Dependencies v2.15
* **Conditions:** Swedish mono · Swedish + Chinese (CTRF) · Swedish + Norwegian (GTRF)
* **Key result:** LAS 25.20 (mono) → 35.56 (+ Chinese) → **49.56 (+ Norwegian)** — typological proximity dominates data volume

\---

### 5\. Figurative Language Detection in Cinematic Dialogue

**Corpus annotation and binary classification**

A joint annotation project labelling movie/TV dialogue as figurative (metaphors, idioms) or literal. Three annotators per sentence with gold-standard resolution via discussion. 14 classifiers benchmarked under TF-IDF; BERT fine-tuning in progress.

* **Dataset:** 517 sentences (419 train / 98 test); \~2:1 class imbalance (literal:figurative)
* **Models:** LinearSVC · Logistic Regression · 12 further classifiers · BERT (`bert-base-uncased`) *(TBD)*
* **Key result:** Best SVM weighted F1 \~0.62; 93.9% of errors are false negatives (figurative predicted as literal)

\---

## More projects coming soon...

\---

## Stack

`PyTorch` · `HuggingFace Transformers` · `scikit-learn` · `XGBoost / LightGBM / CatBoost` · `Universal Dependencies` · `Plotly` · `pandas` / `numpy`

\---

## Author

**Xiaojing Yang**
Uppsala University · Department of Linguistics and Philology

