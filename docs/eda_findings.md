# EDA Findings — ESCI Search Relevance Dataset

**Dataset**: Amazon ESCI (Explicit Semantic Context Information)  
**Source**: `reddit-search-relevance-485717.esci_search` (BigQuery)  
**Notebook**: `notebooks/EDA.ipynb`

---

## 1. Dataset Scale

| Metric | Value |
|---|---|
| Products | 1,802,772 |
| Query-product pairs | 2,621,288 |
| Unique query IDs | 130,652 |
| Unique query strings | 130,193 |
| Locales | 3 (us, jp, es) |
| Average gain (all pairs) | 0.6738 |

**Train / Test split**

| Split | Queries | Pairs |
|---|---|---|
| train | 99,684 | 1,983,272 |
| test | 30,969 | 638,016 |

---

## 2. Relevance Label Distribution

| Label | Name | Count | % of total | Avg gain |
|---|---|---|---|---|
| E | Exact | 1,708,158 | 65.16% | 1.00 |
| S | Substitute | 574,313 | 21.91% | 0.10 |
| C | Complement | 75,652 | 2.89% | 0.01 |
| I | Irrelevant | 263,165 | 10.04% | 0.00 |

**Key observation**: The dataset is 65% Exact — the opposite of most search benchmarks.
ESCI was curated from real Amazon search result pages, which are already pre-filtered by
Amazon's production system. Most shown results are genuinely relevant. This contrasts with
web-crawl datasets where irrelevant results dominate.

**Implication**: The average gain of 0.6738 is high. Hard negative mining via BM25 top-k
is still important — the 10% Irrelevant pairs are structurally the hardest negatives because
they passed initial retrieval despite being irrelevant.

---

## 3. Query Analysis

| Metric | Value |
|---|---|
| Query length p25 | 2 tokens |
| Query length p50 | 3 tokens |
| Query length p75 | 4 tokens |
| Query length p90 | 5 tokens |
| Query length p95 | 6 tokens |
| Coverage with query_maxlen=32 | 100% |

**Observation**: Queries are extremely short — median 3 tokens. This is typical for
e-commerce product search ("red running shoes", "iphone case", "vitamin c supplement").
The ColBERT `query_maxlen=32` setting is more than sufficient; no truncation occurs
for any query in this dataset.

**Products per query**

| Metric | Value |
|---|---|
| Mean | 20.1 |
| Median | 16 |
| p95 | 40 |
| Max | 198 |

**Implication**: BM25 retrieves top 100 candidates — this covers all products for 99%+
of queries. ColBERT then re-ranks those 100, not all 1.8M products.

---

## 4. Product Catalog Quality

**Title length (tokens)**

| Metric | Value |
|---|---|
| Mean | 15.4 |
| p50 | 14 |
| p95 | 31 |
| Coverage with doc_maxlen=128 | 100% |

**Description length (tokens)**

| Metric | Value |
|---|---|
| Mean | 99.6 |
| p50 | 91 |
| p95 | 200 |

**Missing data** (approximate — check figure 07 for exact values)

| Field | Missing % | Status |
|---|---|---|
| title | < 1% | ✓ Complete |
| description | ~60% | ⚠ Major gap |
| bullet_points | ~60% | ⚠ Major gap |
| brand | ~30% | ⚠ Partial |
| color | ~70% | ✗ Mostly absent |

**Implication**: `has_description` (binary) is one of the most important LambdaMART
features precisely because description is missing 60% of the time. When it is
present, it adds meaningful signal. Models must treat description as optional.

---

## 5. Locale & Language

| Locale | Queries | Pairs | Share |
|---|---|---|---|
| us (English) | 97,345 | 1,818,825 | ~69% |
| jp (Japanese) | 18,127 | 446,053 | ~17% |
| es (Spanish) | 15,180 | 356,410 | ~14% |

**Observation**: English dominates but Japanese and Spanish together account for 31%
of the data. Label distribution is consistent across locales — no systematic annotator
bias detected between language groups.

**Implication**: The initial ColBERT model (`bert-base-uncased`) will underperform on
Japanese and Spanish. A multilingual extension should use `xlm-roberta-base` and add
`product_locale` as a LambdaMART categorical feature.

---

## 6. BM25 Baseline Signal

Computed on a 500-query random sample (10,189 pairs). Scores normalised 0–1 within
each query's candidate set.

| Label | Median BM25 score |
|---|---|
| E — Exact | 0.496 |
| C — Complement | 0.410 |
| S — Substitute | 0.245 |
| I — Irrelevant | 0.069 |

**Key observation — unexpected ranking of C vs S**:
Complement products score higher than Substitutes. A complement product (e.g., a phone
case when searching "iPhone 14") often contains the exact product name in its title/description,
boosting its BM25 score. A substitute product (e.g., a different phone model) uses
different vocabulary. BM25 cannot distinguish this semantic relationship — ColBERT can.

**Observation**: BM25 reliably separates Irrelevant (median 0.069) from everything else.
The gap between I and the next-lowest label (S at 0.245) is large, confirming BM25's
value as a hard-negative mining tool.

---

## 7. Query Difficulty

| Metric | Value |
|---|---|
| Queries with zero Exact results | 1 (0.0%) |
| Queries with 100% Irrelevant | 0 (0.0%) |
| Avg % Exact per query | 67.2% |
| Avg % Irrelevant per query | 9.1% |

**Observation**: Almost all queries have at least one Exact result. The dataset is
structurally well-formed — there are very few "dead end" queries where no good
result exists. This means NDCG@10 scores will be interpretable without edge-case
special handling.

---

## 8. Modeling Implications Summary

| Finding | Decision |
|---|---|
| 65% Exact label, not Irrelevant | Hard negatives must be mined deliberately (done: BM25 top-k) |
| query_maxlen=32 covers 100% | Setting is validated; no need to increase |
| doc_maxlen=128 covers 100% of titles | Setting is validated |
| Description missing 60% | `has_description` is a top LambdaMART feature; never treat desc as required |
| BM25 median Irrelevant = 0.069 | BM25 is a strong Irrelevant detector; useful LambdaMART feature |
| Complement > Substitute in BM25 | ColBERT's semantic understanding adds distinct value over BM25 |
| 3 locales, English = 69% | Phase 1 targets English; multilingual extension needs `xlm-roberta` |
| Label mix consistent across splits | No distribution shift; evaluation metrics are reliable |
