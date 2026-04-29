# E-Commerce Search Relevance Engine

A production-grade, 3-stage search relevance pipeline trained on Amazon's ESCI dataset.
Combines BM25 lexical retrieval, ColBERT neural re-ranking, and LambdaMART learning-to-rank fusion — the same architecture class used in large-scale e-commerce search systems.

---

## Results

| Stage | NDCG@10 | MRR@10 | P@10 |
|---|---|---|---|
| BM25 Baseline | 0.6274 | 0.7170 | 0.9195 |
| + ColBERT Re-ranking | 0.7442 | 0.8216 | 0.9390 |
| + LambdaMART Fusion | **0.7553** | **0.8301** | **0.9412** |

ColBERT re-ranking delivers a **+18.6% NDCG@10 improvement** over BM25. LambdaMART fusion adds a further **+1.5%**, bringing the full pipeline to **+20.4% over BM25**.

*Evaluated on 1,000 sampled queries from the ESCI test split.*

---

## Architecture

```
Query
  │
  ▼
BM25 Retrieval (rank-bm25)
  │  Top-100 candidates via lexical matching
  │  ~5 ms
  ▼
ColBERT Re-ranking (bert-base-uncased + MaxSim)
  │  Token-level late interaction across 100 candidates
  │  Query encoder runs at inference; doc embeddings pre-computed
  │  ~50 ms (GPU)
  ▼
LambdaMART Fusion (LightGBM)
  │  10 features: BM25 score, ColBERT score, lexical overlaps, field lengths
  │  Optimises NDCG directly via lambdarank objective
  │  ~1 ms
  ▼
Top-K Results
```

**Why 3 stages?**
BM25 is fast and handles exact lexical matches well but cannot distinguish semantic relationships. ColBERT captures token-level semantics but is too slow to score 1.8M products per query. LambdaMART fuses both signals and learns the optimal combination from labeled data. Each stage filters and re-ranks the previous stage's output, keeping latency under 60 ms end-to-end on GPU.

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
