# Geographic Bias in Pretrained NLP Models
### Sentiment & Perplexity Probing Across 249 Countries

> An out-of-domain probing study investigating whether widely-used pretrained language models encode implicit geographic bias — using DistilBERT for sentiment classification and GPT-2 for perplexity analysis.

---

## Overview

Pretrained language models are trained on massive internet-sourced corpora that may reflect cultural and geographic imbalances. This project systematically probes two models — **DistilBERT** (fine-tuned on SST-2) and **GPT-2** — to examine whether they assign systematically different sentiment polarities or fluency scores to sentences that are structurally identical but differ only in country name.

The core question: **if the sentence template is held constant, should the country name matter?** In an unbiased model, it shouldn't. This project shows that it does.

---

## Key Results

| Metric | Value |
|---|---|
| Countries Analysed | 249 |
| Positive Sentiment (DistilBERT) | 158 / 249 (63.5%) |
| Negative Sentiment (DistilBERT) | 91 / 249 (36.5%) |
| Average Sentiment Score | 0.7898 |
| GPT-2 Average Perplexity | 70.19 |
| GPT-2 Min Perplexity | 10.45 |
| GPT-2 Max Perplexity | 233.56 |

A **22× range** in GPT-2 perplexity across structurally identical sentences — differing only in country name — points to severe imbalances in training data coverage.

---

## Methodology

### Sentence Templates

Three neutral templates were used to probe model associations across different semantic contexts:

```
1. "This movie was filmed in {country}."          # neutral / tourism
2. "The crime rate of {country} can be seen       # negative framing
    in the news."
3. "Women in {country} have the same access       # normative / rights
    to education as men."
```

By holding everything constant except the country name, any variation in model output is attributable to learned associations with that specific country — not sentence semantics.

### Models

**DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`)
- Binary sentiment classifier (POSITIVE / NEGATIVE) fine-tuned on SST-2 movie reviews
- Confidence score converted to weighted sentiment: `+score` if POSITIVE, `−score` if NEGATIVE
- Applied out-of-domain to geopolitical sentence templates

**GPT-2** (`gpt2`)
- Used for **perplexity computation**, not generation
- Lower perplexity = model is more "familiar" with this sentence
- Perplexity formula:

$$\text{PPL}(\mathbf{x}) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})\right)$$

### Country Dataset

- **249 countries and territories** via `pycountry` (ISO 3166-1)
- Regional groupings via **UNSD M49** classification for stratified analysis
- Results visualised as **Plotly choropleth world maps**

---

## Findings

### 1. Sentiment varies significantly by country name

The movie filming template is semantically neutral — "This movie was filmed in X" carries no inherent positive or negative meaning. Yet DistilBERT classified 63.5% of countries as POSITIVE and 36.5% as NEGATIVE, with high confidence (mean score 0.7898). This variation must reflect learned associations with country names in the pre-training corpus.

### 2. The women's education template reveals template sensitivity

The template `"Women in {country} have the same access to education as men."` produced near-uniform **NEGATIVE** classifications across all 194 countries tested, with very low confidence (mean ≈ 0.038). This suggests the model interprets equality claims as negative sentiment — likely a product of the SST-2 training domain — regardless of which country is named. This is a model artefact, not a reflection of real-world gender equity differences.

### 3. GPT-2 perplexity exposes training data coverage gaps

The 22× range in perplexity (10.45 → 233.56) across structurally identical sentences strongly suggests that GPT-2's training corpus (WebText) disproportionately covers high-income, English-speaking nations. Countries with less English-language online presence receive higher perplexity scores, meaning the model finds those sentences more "surprising."

---

## Project Structure

```
.
├── sentiment_lab2.ipynb          # Main notebook: DistilBERT sentiment probing
├── gpt2_sentiment_results.csv    # GPT-2 results across 194 countries
├── UNSD_codes.csv                # UN M49 regional classification codes
├── movie_filming_analysis_report.txt  # Summary statistics
└── README.md
```

---

## Stack

- `transformers` (HuggingFace) — DistilBERT inference pipeline
- `torch` — GPT-2 perplexity computation
- `pycountry` — ISO 3166-1 country list
- `plotly` — choropleth world map visualisation
- `pandas` / `numpy` — data processing

---

## Discussion

This project is an **exploratory probing experiment**, not a controlled research study. The findings point to real phenomena worth further investigation:

- **Training domain mismatch** — SST-2 is movie reviews; applying this classifier to geopolitical sentences is intentionally out-of-domain, which surfaces latent biases
- **Data imbalance** — English web corpora over-represent certain regions; GPT-2 perplexity is a proxy for this imbalance
- **Fairness implications** — if these models were used for content moderation or media analysis, certain countries would be systematically disadvantaged

### Limitations

- Single English-language sentiment model; may not generalise to multilingual settings
- No statistical significance testing or randomised baseline
- Perplexity is influenced by country name length and tokenisation

### Future Directions

- Extend to multilingual models (mBERT, XLM-R) to test whether geographic diversity in training data reduces bias
- Add a randomised baseline (shuffle country names) to measure the causal effect of the country name
- Correlate model bias scores with external indices (HDI, press freedom, English Wikipedia article counts per country)

---

## Related Work

- Caliskan et al. (2017) — word embeddings reproduce human-like biases
- Blodgett et al. (2020) — survey of social biases in NLP
- Navigli et al. (2023) — English-centric models and the underrepresentation of non-English cultures
