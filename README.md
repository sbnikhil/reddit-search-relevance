# Search Relevance Engine

End-to-end product search pipeline trained on Amazon's ESCI dataset (1.8M products, 130K queries).
Two novel contributions: **ρ-conditioned rationale distillation** for cross-encoder training, and the **first honest end-to-end evaluation** on TREC Product Search 2023 — exposing how much published ESCI benchmarks overstate real retrieval quality.

---

## Results

### Re-ranking evaluation (ESCI editorial pool, Protocol A)
*Comparable to published ESCI papers — pre-selected candidates, ~20 per query*

| System | NDCG@10 | MRR@10 |
|---|---|---|
| BM25 Baseline | 0.6329 | 0.7077 |
| ColBERT Re-ranking | **0.7622** | **0.8312** |
| LambdaMART Fusion | 0.7544 | 0.8270 |
| Cross-encoder (in progress) | TBD | TBD |

> **Finding:** LambdaMART degrades ColBERT by −0.008 NDCG@10. Root cause: a single LambdaMART
> trained on all queries learns unstable feature weights when BM25 and ColBERT rankings are
> orthogonal (low Spearman ρ). Documented in `results/ltr_regression_analysis.json`.
> Cross-encoder replaces LambdaMART in the new architecture.

*NDCG computed with gain=[0,1,3,7] for [I/C, S, E] matching LambdaMART's training objective.*

### End-to-end retrieval (TREC Product Search 2023, Protocol B)
*First measurement retrieving from the full 1.8M catalog — honest evaluation*

| System | NDCG@10 | Note |
|---|---|---|
| BM25 | TBD | Establishing baseline |
| BM25 + FAISS hybrid | TBD | Dense + sparse |
| Full pipeline | TBD | Hybrid + cross-encoder |

---

## Architecture

```
Query
  │
  ▼ [Query Router — rule-based, 0ms]
  │  ≤2 or 3-4 tokens → BM25 top-50 + FAISS top-50 → RRF → top-100
  │  ≥5 tokens       → BM25 top-100 only (long queries are already specific)
  │
  ▼ [Stage 1: Hybrid Retrieval]
  │  BM25 (rank-bm25, 1.8M products)  ~5ms
  │  FAISS IVFFlat (ColBERT [CLS] embeddings, dim=128, n_centroids=2048)  ~2ms
  │  Reciprocal Rank Fusion (k=60)
  │
  ▼ [Stage 2: Cross-Encoder Re-ranking]
  │  ModernBERT-base backbone
  │  Input: [CLS] query [SEP] title [SEP] description
  │  Output: P(E) + 0.1·P(S) + 0.01·P(C)
  │  Trained with ρ-conditioned rationale distillation
  │
  ▼ Top-20 results
```

**One model per role:** ColBERT = dense retriever (FAISS index only). Cross-encoder = re-ranker.
LambdaMART kept for ablation comparison, not in the production path.

---

## Dataset

**Amazon ESCI** (Explicit Semantic Context Information) — a real Amazon production dataset with 4-level graded relevance labels assigned by human annotators.

| Metric | Value |
|---|---|
| Products | 1,802,772 |
| Query-product pairs | 2,621,288 |
| Unique queries | 130,652 |
| Locales | English (69%), Japanese (17%), Spanish (14%) |
| Train / Test | 99,684 / 30,969 queries |

**Relevance labels:**

| Label | Meaning | Gain | % of dataset |
|---|---|---|---|
| E — Exact | Direct answer to the query | 1.00 | 65.2% |
| S — Substitute | Similar but not exact | 0.10 | 21.9% |
| C — Complement | Related accessory | 0.01 | 2.9% |
| I — Irrelevant | Not useful | 0.00 | 10.0% |

The dataset is 65% Exact because it was sampled from Amazon's production search results — already pre-filtered by Amazon's own system. This is the opposite of web-crawl datasets. Hard negatives are deliberately mined to compensate.

**BM25 signal quality (500-query sample):**

| Label | Median BM25 score |
|---|---|
| Exact | 0.496 |
| Complement | 0.410 |
| Substitute | 0.245 |
| Irrelevant | 0.069 |

Complement products score higher than Substitutes on BM25 because they often contain the exact query product name in their title (e.g., "iPhone 14 case" when searching "iPhone 14"). ColBERT's semantic understanding resolves this ambiguity.

---

## Key Design Decisions

**ColBERT over bi-encoders (Sentence-BERT)**
E-commerce queries are short and ambiguous (median 3 tokens). ColBERT's token-level MaxSim interaction captures partial matches that a single dense vector misses — "running shoes" matching "lightweight trail running athletic shoe" token by token rather than as a single compressed representation.

**LambdaMART over a cross-encoder**
A cross-encoder (BERT scoring query+document jointly) would be more accurate but requires a forward pass per candidate. LambdaMART adds near-zero latency while learning to combine BM25 and ColBERT signals, including engineered features the neural model doesn't see (field-level overlaps, brand match, description presence).

**Hard negative mining**
With 65% Exact labels, random negatives would mostly be other Exact matches — uninformative for training. BM25 top-k mining retrieves products that look relevant lexically but are judged Irrelevant or Substitute, forcing the model to learn the distinction.

**LambdaMART label mapping**
ESCI float gains (1.0, 0.1, 0.01, 0.0) are mapped to integer levels (3, 2, 1, 0) with `label_gain=[0, 1, 3, 7]` matching the NDCG gain formula 2^r − 1. This ensures the objective directly optimises the evaluation metric.

---

## Known Limitations

**Multilingual gap:** `bert-base-uncased` is English-only. Japanese (17%) and Spanish (14%) queries will underperform. The fix is switching the backbone to `xlm-roberta-base`, which handles all three locales. This is scoped as a follow-up.

**ColBERT initialisation:** The model is trained from a general BERT checkpoint rather than an IR-pretrained checkpoint (e.g., ColBERT-v2 pre-trained on MS MARCO). Starting from MS MARCO would likely give better initial token representations and require fewer training epochs.

**Re-encoding at serving time:** Document embeddings for BM25 top-100 candidates are re-encoded at query time rather than pre-computed. At batch size 100 on GPU this takes ~50 ms — acceptable, but a proper vector index (Vertex AI Matching Engine, FAISS) would reduce this to lookup latency for larger catalogs.

---

## Tech Stack

| Component | Tool | Role |
|---|---|---|
| First-stage retrieval | rank-bm25 | Lexical candidate generation |
| Neural re-ranker | ColBERT (BERT + MaxSim) | Token-level late interaction scoring |
| LTR fusion | LightGBM LambdaMART | NDCG-optimised feature fusion |
| Training infra | Vertex AI Custom Jobs | GPU training (T4) |
| Data warehouse | BigQuery | ESCI dataset + ColBERT score storage |
| Serving | FastAPI + uvicorn | REST endpoint |
| Containerisation | Docker + Cloud Run | Deployment |
| CI | GitHub Actions | Test suite + NDCG regression gate |

---

## Running the Pipeline

```bash
# Setup
bash setup.sh && source venv/bin/activate
cp .env.example .env  # fill in GCP credentials

# 1. Load ESCI data into BigQuery
python scripts/01_load_bigquery.py

# 2. Compute BM25 baseline
python scripts/02_bm25_baseline.py

# 3. Mine hard negatives for ColBERT training
python scripts/03_mine_hard_negatives.py

# 4. Train ColBERT on Vertex AI (T4 GPU, ~16 hrs for 5 epochs)
python scripts/04_train_colbert.py --submit

# 5. Generate ColBERT scores for all pairs (resumable)
python scripts/08_generate_colbert_scores.py

# 6. Train LambdaMART with real ColBERT features
python scripts/05_train_lambdamart.py

# 7. Evaluate all 3 stages
python scripts/06_evaluate.py

# 8. Deploy serving endpoint
python scripts/07_deploy_vertex.py
```

## Local Serving

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8080
# or
docker-compose up
```

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "noise cancelling headphones", "top_k": 5}'
```

## Tests

```bash
pytest tests/ -v  # 34 tests across ColBERT, LambdaMART, and serving layers
```

---

## EDA

Key findings from exploratory analysis are documented in [docs/eda_findings.md](docs/eda_findings.md).
Full analysis with figures: [notebooks/EDA.ipynb](notebooks/EDA.ipynb).
