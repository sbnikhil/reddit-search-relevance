# FINAL PLAN — Search Relevance Engine
## For coding agent execution | 2-day sprint | June 2026

---

## SECTION 0: WHERE WE STAND RIGHT NOW

### What exists and is VALID — keep everything below

| Asset | Location | Status | Keep? |
|---|---|---|---|
| ColBERT model (bert-base-uncased, dim=128, [Q]/[D] tokens) | GCS: search-models-nikhil | Trained 5 epochs | ✅ Keep as dense retriever for FAISS |
| 2.68M ColBERT scores | BigQuery: esci_search | Complete, 130K queries | ✅ Use for FAISS index + training |
| BM25 index (rank-bm25, 1.8M products) | Built at serving startup | Working | ✅ Keep as sparse retriever |
| Three conditioned LambdaMART models | GCS: search-models-nikhil | Trained per ρ-bin | ⚠️ Keep but do NOT use as final ranker — evaluate only |
| Spearman ρ analysis (script 10) | results/ltr_regression_analysis.json | Complete | ✅ Critical input to Contribution 1 |
| Hard negative triplets (script 03) | GCS | Complete | ✅ Reuse for cross-encoder training |
| ESCI data in BigQuery | esci_search.{products, queries, labels} | Complete | ✅ Training data |
| FastAPI serving layer | serving/app.py | Working, single process | ✅ Refactor, don't rewrite |
| Scripts 12, 13, 14 | Partially done | Incomplete | ⚠️ See below |
| CI pipeline | .github/workflows/ci.yml | Broken (static JSON gate) | 🔴 Fix Day 1 |

### What to REMOVE / DISCARD immediately

| Thing | Reason |
|---|---|
| LambdaMART as the final production ranker | Cross-encoder makes it redundant. It also hurts ColBERT on corrected eval. |
| LLM query expansion for short queries | 200-500ms latency. Non-starter for production. Not impressive. |
| Post-reranking Complement filter | Wrong stage. Cross-encoder already handles this via training labels. |
| Val NDCG numbers 0.81-0.87 from script 11 | Wrong eval protocol. Incomparable to everything. |
| README claim that LambdaMART NDCG = 0.7553 | Wrong. Actual is 0.7332. Fix Day 1. |
| Tiered serving architecture | No production traffic data to classify query tiers. Cosmetic. |
| CPR (Complement Pollution Rate) metric | Does not exist in literature. Use Complement-P@10 instead if needed. |
| Comparison to KDD Cup NDCG in main results table | Different task (re-ranking vs retrieval). Incomparable. Cite in related work only. |

### The two contributions that are real and worth building

**Contribution 1:** ρ-conditioned rationale distillation — use LLM reasoning supervision
weighted by BM25/ColBERT retriever disagreement to train a cross-encoder.
Novel because: rationale distillation on ESCI exists (COLING 2025), ρ-conditioned
LTR exists (your script 10), the combination does not exist anywhere.

**Contribution 2:** Honest end-to-end retrieval evaluation — first measurement of
what ESCI-trained models actually achieve when retrieving from the full 1.8M catalog,
evaluated on TREC Product Search 2023 (which was built specifically because ESCI
lacks end-to-end retrieval evaluation).

Everything else supports these two. Nothing else is a contribution.

---

## SECTION 1: ARCHITECTURE (FINAL, LOCKED)

```
Query
  │
  ▼ [Query Router — pre-retrieval, rule-based, zero LLM calls]
  │  token_count <= 2 → synonym expansion (static dict, 0ms overhead)
  │  token_count 3-4  → standard
  │  token_count >= 5 → BM25 only, skip FAISS
  │
  ▼
[Stage 1: Hybrid Retrieval]
  BM25 top-50 (rank-bm25, 1.8M products)
  +
  FAISS ANN top-50 (ColBERT [CLS] embeddings, IVFFlat, dim=128)
  ──── merged via Reciprocal Rank Fusion (k=60) ────────────────
  top-100 candidates (deduplicated)
  │
  ▼
[Stage 2: Cross-Encoder Re-ranking]
  RexBERT (answerdotai/ModernBERT-base as fallback) cross-encoder
  Input: [CLS] query [SEP] title [SEP] description
  Output: P(E) + 0.1*P(S) + 0.01*P(C) — matches ESCI gain formula
  Trained with ρ-conditioned rationale distillation
  Returns top-20
  │
  ▼
Final ranked list
```

**One model per role. No ambiguity:**
- ColBERT = dense retriever only (FAISS index)
- Cross-encoder = re-ranker (replaces ColBERT MaxSim scoring AND LambdaMART)
- LambdaMART = evaluated for comparison only, not in production path

---

## SECTION 2: EVALUATION STRATEGY (FINAL, LOCKED)

### Three evaluation protocols, each for a different purpose:

**Protocol A — Re-ranking (ESCI editorial pool)**
- Use: Compare cross-encoder variants against each other and against COLING 2025
- Candidates: ESCI pre-selected pool (~20 per query)
- Labels: ESCI human labels (100% coverage)
- Metric: NDCG@10 with gain=[0,1,3,7], ROC-AUC for classification
- Comparable to: KDD Cup 2022 numbers (note this clearly in paper/README)
- Script: scripts/06_evaluate.py --protocol editorial_pool

**Protocol B — End-to-end retrieval (TREC Product Search 2023)**
- Use: Honest end-to-end retrieval evaluation
- Candidates: Retrieved by your system from full 1.8M catalog
- Labels: TREC 2023 qrels (proper full-catalog judgments, 998 queries)
- Metric: NDCG@10, Recall@100 for Exact-labeled products
- Comparable to: Nothing published yet — you are establishing the baseline
- Script: scripts/06_evaluate.py --protocol trec2023

**Protocol C — Ablation (ESCI editorial pool, cross-encoder variants only)**
- Use: Isolate what each component contributes
- Same as Protocol A but run 4 model variants
- Script: scripts/06_evaluate.py --protocol ablation

### One NDCG formulation everywhere — no exceptions:
```python
GAIN_MAP = {'E': 3, 'S': 1, 'C': 0, 'I': 0, None: 0}
# NDCG uses 2^gain - 1: {0:0, 1:1, 3:7}
# This matches LambdaMART's label_gain=[0,1,3,7]
# Apply this in scripts 06, 09, 10, 11 — delete all other formulations
```

---

## SECTION 3: DAY-BY-DAY EXECUTION PLAN

### DAY 1 (8 hours): Fix everything broken, establish honest baseline

#### Hour 1: Four immediate fixes (do these first, in order)

**Fix 1 — README**
Open README.md. Find results table. Replace:
- LambdaMART NDCG@10: 0.7553 → 0.7332
- LambdaMART MRR@10: whatever is wrong → 0.8046
Add sentence under results table:
"Note: LambdaMART degrades ColBERT by -0.011 NDCG@10.
Root cause and fix described in docs/FINDINGS.md (in progress)."

**Fix 2 — CI gate**
In .github/workflows/ci.yml, replace the static JSON read:
```yaml
- name: NDCG regression gate
  run: |
    python scripts/06_evaluate.py \
      --protocol editorial_pool \
      --sample 200 \
      --output /tmp/ci_eval.json
    python -c "
    import json, sys
    with open('/tmp/ci_eval.json') as f:
        r = json.load(f)
    threshold = 0.70
    ndcg = r['ndcg_at_10']
    print(f'NDCG@10: {ndcg:.4f} (threshold: {threshold})')
    sys.exit(0 if ndcg >= threshold else 1)
    "
```
Note: set threshold after you have honest baseline numbers.

**Fix 3 — GCP project name**
In every script where PROJECT_ID = "reddit-search-relevance-485717":
Replace with: PROJECT_ID = os.getenv("GCP_PROJECT_ID", "search-relevance-500100")
Add to config/__init__.py: PROJECT_ID = os.getenv("GCP_PROJECT_ID", "search-relevance-500100")

**Fix 4 — Unified NDCG function**
In utils/metrics.py, ensure ONE implementation exists:
```python
GAIN_MAP = {'E': 3, 'S': 1, 'C': 0, 'I': 0, None: 0}

def ndcg_at_k(ranked_labels, k=10):
    """
    ranked_labels: list of ESCI label strings or None (unlabeled)
    Uses gain=[0,1,3,7] matching LambdaMART objective.
    2^level - 1 formula: level 0→0, level 1→1, level 2→3, level 3→7
    """
    gains = {0: 0, 1: 1, 2: 3, 3: 7}
    
    def dcg(labels, k):
        score = 0.0
        for i, label in enumerate(labels[:k]):
            level = GAIN_MAP.get(label, 0)
            score += gains[level] / np.log2(i + 2)
        return score
    
    actual = dcg(ranked_labels, k)
    ideal_labels = sorted(ranked_labels, key=lambda l: GAIN_MAP.get(l, 0), reverse=True)
    ideal = dcg(ideal_labels, k)
    return actual / ideal if ideal > 0 else 0.0
```
Delete all other NDCG implementations. Search for "ndcg" across all scripts
and replace with this function.

#### Hours 2-3: Download TREC Product Search 2023 data

```bash
# Download TREC 2023 qrels
wget https://trec.nist.gov/data/product/2023-product-qrels.txt -O data/trec2023_qrels.txt
wget https://trec.nist.gov/data/product/2023-test-queries.tsv -O data/trec2023_queries.tsv

# Verify format:
# qrels format: query_id 0 product_id relevance_level
# queries format: query_id \t query_text
# TREC uses same ESCI product catalog — product_ids are ASINs, same as your BigQuery table

head -5 data/trec2023_qrels.txt
head -5 data/trec2023_queries.tsv
```

Verify the product IDs in TREC qrels exist in your BigQuery products table.
Run: SELECT COUNT(*) FROM esci_search.products WHERE product_id IN (
  SELECT product_id FROM trec_qrels)
If > 80% match: proceed. If < 80%: flag and report — may need product ID mapping.

#### Hours 3-5: Rewrite scripts/06_evaluate.py

```python
# scripts/06_evaluate.py
"""
Unified evaluation script supporting three protocols.
Usage:
  python scripts/06_evaluate.py --protocol editorial_pool [--sample N]
  python scripts/06_evaluate.py --protocol trec2023
  python scripts/06_evaluate.py --protocol ablation
"""

import argparse
from utils.metrics import ndcg_at_k
from utils.gcs import download_model

def load_editorial_pool(sample=None):
    """Load ESCI test split from BigQuery. Already-labeled candidates only."""
    # Query: SELECT query_id, query, product_id, esci_label, product_title,
    #        product_description FROM esci_search.examples
    #        WHERE split='test' AND large_version=1
    # Returns: dict {query_id: {'query': str, 'candidates': [(product_id, label)]}}
    pass

def load_trec2023():
    """Load TREC 2023 queries and qrels."""
    # Read data/trec2023_queries.tsv → {query_id: query_text}
    # Read data/trec2023_qrels.txt → {query_id: {product_id: relevance_level}}
    # relevance levels in TREC: map to ESCI equivalents
    # TREC level 3 → E, level 2 → S, level 1 → C, level 0 → I
    pass

def retrieve_bm25_top100(query, bm25_index, product_lookup):
    """BM25 retrieval from full 1.8M catalog."""
    scores = bm25_index.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:100]
    return [(product_lookup[i]['product_id'], scores[i]) for i in top_idx]

def retrieve_hybrid_top100(query, bm25_index, faiss_index, query_encoder, product_lookup):
    """BM25 + FAISS hybrid retrieval."""
    bm25_candidates = retrieve_bm25_top100(query, bm25_index, product_lookup)
    query_emb = query_encoder.encode_query(query)
    _, faiss_idx = faiss_index.search(query_emb.reshape(1, -1), 50)
    faiss_candidates = [(product_lookup[i]['product_id'], 1.0/(rank+1))
                        for rank, i in enumerate(faiss_idx[0])]
    return reciprocal_rank_fusion([bm25_candidates, faiss_candidates])

def evaluate(queries, label_lookup, retriever, reranker, k=10):
    results = []
    for query_id, query_text in queries.items():
        # Retrieve
        candidates = retriever(query_text)
        
        # Label candidates (None if not in label_lookup)
        labeled = [(pid, label_lookup.get(query_id, {}).get(pid, None))
                   for pid, _ in candidates]
        
        # Coverage: what fraction have labels
        n_labeled = sum(1 for _, l in labeled if l is not None)
        coverage = n_labeled / len(labeled)
        
        # Re-rank if reranker provided
        if reranker:
            reranked_labels = reranker.rerank_and_get_labels(query_text, labeled)
        else:
            reranked_labels = [l for _, l in labeled]
        
        ndcg = ndcg_at_k(reranked_labels, k=k)
        results.append({
            'query_id': query_id,
            'ndcg': ndcg,
            'coverage': coverage,
        })
    
    return {
        'ndcg_at_10': np.mean([r['ndcg'] for r in results]),
        'mean_coverage': np.mean([r['coverage'] for r in results]),
        'n_queries': len(results),
        'per_query': results
    }
```

#### Hours 5-6: Run baseline evaluation — get honest numbers

```bash
# Run three baselines:

# 1. BM25 on editorial pool (replicates your current numbers, now with correct NDCG)
python scripts/06_evaluate.py --protocol editorial_pool --retriever bm25 --reranker none

# 2. ColBERT on editorial pool (your current best model)
python scripts/06_evaluate.py --protocol editorial_pool --retriever bm25 --reranker colbert

# 3. BM25 end-to-end on TREC2023 (establishes your honest baseline)
python scripts/06_evaluate.py --protocol trec2023 --retriever bm25 --reranker none

# Save all outputs to results/baselines.json
# These are the numbers every subsequent result is compared against
```

Record these numbers. Do not proceed until you have them.
Expected: TREC numbers will be lower than editorial pool numbers. That's correct.

#### Hours 6-7: Build FAISS index

```python
# scripts/15_build_faiss_index.py

"""
Builds FAISS IVFFlat index from ColBERT document embeddings.

Input: ColBERT [CLS] embeddings for all products with ESCI labels
       (from BigQuery table esci_search.colbert_scores — script 08 output)
       NOTE: Use the [CLS] token embedding, not the full token matrix.
       If script 08 stored full token embeddings, extract [:, 0, :] (first token).

Output: /tmp/product_index.faiss → GCS: search-models-nikhil/faiss/product_index.faiss

Index specs:
  - Dimension: 128 (your ColBERT projection dimension)
  - Type: IndexIVFFlat with IndexFlatIP quantizer
  - n_centroids: 2048 (appropriate for ~200K labeled products)
  - metric: INNER_PRODUCT (cosine similarity after L2 normalization)
  - nprobe: 32 at query time

IMPORTANT: Index only the ~200K products that have ESCI labels.
Not the full 1.8M — we don't have embeddings for unlabeled products.
The FAISS index supplements BM25 for semantic recall within labeled products.
The BM25 index covers all 1.8M for lexical recall.
This is the correct hybrid: dense for semantic + sparse for lexical.
"""

import faiss
import numpy as np
from google.cloud import bigquery

def build_index():
    client = bigquery.Client()
    
    # Load embeddings from BigQuery
    # script 08 stored colbert scores — need the raw embeddings
    # If only scores were stored (not embeddings), you need to re-encode
    # Check your BigQuery table schema first:
    # SELECT column_name FROM information_schema.columns
    # WHERE table_name = 'colbert_scores'
    
    query = """
    SELECT product_id, colbert_embedding
    FROM esci_search.colbert_scores
    WHERE colbert_embedding IS NOT NULL
    LIMIT 200000
    """
    # If embeddings not stored, encode products using existing ColBERT model:
    # load model from GCS, encode product titles+descriptions in batches of 64

    embeddings = np.array([row.colbert_embedding for row in results]).astype('float32')
    product_ids = [row.product_id for row in results]
    
    # L2 normalize (required for inner product = cosine similarity)
    faiss.normalize_L2(embeddings)
    
    d = embeddings.shape[1]  # 128
    n_centroids = 2048
    
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_centroids, faiss.METRIC_INNER_PRODUCT)
    
    # Train on all embeddings (need at least 39*n_centroids = 79K samples)
    print(f"Training index on {len(embeddings)} embeddings...")
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 32
    
    print(f"Index built: {index.ntotal} vectors")
    
    # Save locally then upload to GCS
    faiss.write_index(index, '/tmp/product_index.faiss')
    
    # Save product_id mapping (needed to convert FAISS indices back to product_ids)
    import json
    with open('/tmp/product_id_map.json', 'w') as f:
        json.dump(product_ids, f)
    
    # Upload both to GCS
    # gsutil cp /tmp/product_index.faiss gs://search-models-nikhil/faiss/
    # gsutil cp /tmp/product_id_map.json gs://search-models-nikhil/faiss/

build_index()
```

```bash
# Run it. Expected time: 10-20 minutes.
python scripts/15_build_faiss_index.py

# Verify: search 5 test queries, check top-10 results look reasonable
python -c "
import faiss, json, numpy as np
index = faiss.read_index('/tmp/product_index.faiss')
index.nprobe = 32
# Encode a test query with ColBERT query encoder
# Check results look semantically relevant
"
```

#### Hour 7-8: Run hybrid retrieval baseline

```bash
# Add hybrid retriever to script 06 and run:
python scripts/06_evaluate.py --protocol editorial_pool --retriever hybrid --reranker none
python scripts/06_evaluate.py --protocol trec2023 --retriever hybrid --reranker none

# Compare recall@100 for Exact-labeled products:
# BM25 only recall@100 vs Hybrid recall@100
# If hybrid > BM25 on Exact recall: FAISS is contributing. Good.
# If hybrid ≈ BM25: FAISS isn't adding much on this dataset. Document it.
```

Save all Day 1 results to results/day1_baselines.json.
**Day 1 ends here. You now have honest baselines for everything.**

---

### DAY 2 (8 hours): Build the cross-encoder, ablation, finalize

#### Hours 1-3: Generate LLM rationales (start this first, runs in background)

```python
# scripts/16_generate_rationales.py
"""
Generates LLM reasoning chains for hard training pairs.
Uses Claude API (you have access via Anthropic).
Cost estimate: 20K pairs * ~400 tokens output * $3/1M tokens = ~$24

Which pairs to generate rationales for:
  Priority 1: All C-labeled pairs (only ~75K in training set — use ALL)
  Priority 2: Low-ρ query pairs with S labels (disagreement cases)
  Priority 3: E/I pairs where BM25 score disagrees with label
    (E labeled but low BM25 score = ColBERT needed here)
    (I labeled but high BM25 score = BM25 fooled, hard case)

Target: 20K pairs total (not 50K — cost and time reduction)
Rationale quality > quantity here.

Filter: Only keep rationales where:
  LLM predicted label == human ESCI label (discard contradictions)
  This ensures no contradictory training signal

Output format (one JSON per line in data/rationales.jsonl):
{
  "query_id": "...",
  "product_id": "...",
  "query": "...",
  "title": "...",
  "description": "...",
  "esci_label": "C",
  "llm_soft_labels": [0.05, 0.10, 0.75, 0.10],  # [P(E), P(S), P(C), P(I)]
  "rationale": "1. Query Intent: ...\n2. Product: ...\n5. COMPLEMENT because..."
}
"""

import anthropic
import json
from google.cloud import bigquery

PROMPT = """You are an expert in e-commerce search relevance.

Query: "{query}"
Product Title: "{title}"
Product Description: "{description}"
Human Label: {label} ({label_definition})

Analyze this query-product pair:

1. QUERY INTENT: What is the user looking for? What attributes matter most?
2. PRODUCT IDENTITY: What is this product? What category?
3. CATEGORY MATCH: Same category as query intent? Yes/No and why.
4. ATTRIBUTE MATCH: Do product specs match query requirements?
5. VERDICT: Assign probabilities summing to 1.0:
   P(Exact)=X, P(Substitute)=X, P(Complement)=X, P(Irrelevant)=X
   Then state which label you assign and why.

Output JSON only:
{{"rationale": "your analysis", "soft_labels": {{"E": 0.X, "S": 0.X, "C": 0.X, "I": 0.X}}, "predicted_label": "E/S/C/I"}}"""

LABEL_DEFS = {
    'E': 'product directly satisfies the query',
    'S': 'reasonable alternative but misses some requirements',
    'C': 'related but serves a different primary function (accessory)',
    'I': 'does not satisfy the query'
}

client = anthropic.Anthropic()

def generate_rationale(query, title, description, label):
    prompt = PROMPT.format(
        query=query, title=title,
        description=(description or 'No description available')[:500],
        label=label, label_definition=LABEL_DEFS[label]
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        result = json.loads(response.content[0].text)
        # Only keep if LLM agrees with human label
        if result['predicted_label'] == label:
            return result
        return None  # Discard contradictions
    except:
        return None

# Load hard pairs from BigQuery, generate, save to data/rationales.jsonl
# Use asyncio or ThreadPoolExecutor for parallel API calls (respect rate limits)
# Target: 20K valid rationales
```

**Start this script running immediately at the start of Day 2. It takes 2-3 hours.**
While it runs, proceed with the cross-encoder in parallel.

#### Hours 1-4 (parallel with rationale generation): Build cross-encoder

```python
# models/cross_encoder.py

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESCICrossEncoder(nn.Module):
    """
    Cross-encoder for ESCI relevance scoring.
    
    Backbone priority:
    1. answerdotai/ModernBERT-base (check HuggingFace first)
    2. microsoft/deberta-v3-base (fallback if ModernBERT unavailable)
    3. bert-base-uncased (last resort)
    
    Input: "[CLS] query [SEP] product_title [SEP] product_description [SEP]"
    Max length: 512 tokens. Truncate description if needed.
    
    Output: relevance score = P(E) + 0.1*P(S) + 0.01*P(C)
    This matches the ESCI NDCG gain formula exactly.
    """
    
    BACKBONES = [
        "answerdotai/ModernBERT-base",
        "microsoft/deberta-v3-base",
        "bert-base-uncased"
    ]
    
    def __init__(self):
        super().__init__()
        for backbone in self.BACKBONES:
            try:
                self.encoder = AutoModel.from_pretrained(backbone)
                self.tokenizer = AutoTokenizer.from_pretrained(backbone)
                self.backbone_name = backbone
                print(f"Loaded backbone: {backbone}")
                break
            except Exception as e:
                print(f"Could not load {backbone}: {e}, trying next...")
        
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, 4)  # E, S, C, I
    
    def encode_pair(self, query, title, description, max_length=512):
        # Truncate description to leave room for query and title
        text = f"{query} {self.tokenizer.sep_token} {title} {self.tokenizer.sep_token} {description or ''}"
        return self.tokenizer(
            text, max_length=max_length, truncation=True,
            padding='max_length', return_tensors='pt'
        )
    
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        logits = self.classifier(cls)
        probs = F.softmax(logits, dim=-1)
        # Score matches NDCG gain: E contributes most, S less, C minimal, I zero
        score = probs[:, 0] + 0.1 * probs[:, 1] + 0.01 * probs[:, 2]
        return score, logits, probs
```

```python
# models/rho_distillation.py

def rho_conditioned_loss(student_logits, hard_labels, llm_soft_labels,
                          query_rho, base_alpha=0.3):
    """
    ρ-conditioned rationale distillation loss.
    
    Standard distillation (COLING 2025): uniform alpha=0.3 for all pairs.
    Our contribution: alpha varies by retriever agreement.
    
    alpha(ρ) = base_alpha + (1 - ρ) * (1 - base_alpha)
    ρ=0 (max disagreement): alpha=1.0 → pure rationale supervision
    ρ=1 (perfect agreement): alpha=0.3 → mostly hard labels
    
    Intuition: When BM25 and ColBERT disagree, the case is hard.
    LLM reasoning adds most value on hard cases.
    When they agree, hard labels are sufficient.
    
    Args:
        student_logits: (B, 4) cross-encoder output
        hard_labels: (B,) integer ESCI levels {0,1,2,3}
        llm_soft_labels: (B, 4) LLM probability distribution [P(E),P(S),P(C),P(I)]
                         For pairs without rationales: use one-hot from hard label
        query_rho: (B,) Spearman ρ for each pair's query
                   Precomputed from script 10 results, joined to training data
    """
    alpha = base_alpha + (1 - query_rho) * (1 - base_alpha)  # (B,)
    
    # Hard label loss
    ce_loss = F.cross_entropy(student_logits, hard_labels, reduction='none')  # (B,)
    
    # Soft label loss (KL divergence)
    kl_loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        llm_soft_labels.clamp(min=1e-8),
        reduction='none'
    ).sum(dim=-1)  # (B,)
    
    # Combine with per-sample ρ-conditioned weighting
    loss = (1 - alpha) * ce_loss + alpha * kl_loss
    return loss.mean()
```

#### Hours 3-5: Training script with four ablation variants

```python
# scripts/17_train_cross_encoder.py
"""
Trains 4 cross-encoder variants for ablation study.
Run on Vertex AI T4 GPU. Estimated time: 2-3 hours per variant.
Submit all 4 as parallel Vertex AI jobs.

VARIANT 1: Vanilla cross-encoder
  - Loss: cross-entropy on hard ESCI labels only
  - No rationale data
  - No ρ-conditioning
  - Baseline: establishes your own cross-encoder baseline

VARIANT 2: Rationale distillation (uniform alpha=0.3)
  - Loss: 0.7 * CE + 0.3 * KL(student, llm_soft_labels)
  - Uses rationale data for pairs that have it
  - Pairs without rationales: soft_label = one-hot from hard label (alpha=0.3 still applied)
  - Replicates COLING 2025 approach on your data

VARIANT 3: ρ-weighting without rationale (control)
  - Loss: ρ-conditioned CE only (no soft labels)
  - weight(ρ) applied to CE loss: low-ρ pairs get higher loss weight
  - Tests whether ρ-weighting alone helps, independent of rationale
  - This isolates the ρ effect from the distillation effect

VARIANT 4: ρ-conditioned rationale (full method — novel contribution)
  - Loss: rho_conditioned_loss() as defined above
  - Uses rationale data where available
  - ρ gates how much rationale supervision each pair receives

Training config (same for all variants):
  optimizer: AdamW
  lr: 2e-5
  warmup: 10% of steps
  batch_size: 32
  epochs: 3
  gradient_clipping: 1.0
  
Data split:
  Training: ESCI train split (large version)
  Validation: ESCI dev split (early stopping on NDCG@10)
  Test: Run evaluation via scripts/06_evaluate.py separately

For each query in training data, attach:
  query_rho: from results/ltr_regression_analysis.json
  If query not in ρ analysis: use mean ρ = 0.425 as fallback
  
For each (query, product) pair in training data, attach:
  llm_soft_label: from data/rationales.jsonl if available
  If not available: [1,0,0,0] for E, [0,1,0,0] for S, etc. (one-hot)
"""

VARIANT_CONFIGS = {
    'vanilla': {
        'use_rationales': False,
        'use_rho_conditioning': False,
        'alpha': 0.0  # pure CE
    },
    'rationale_uniform': {
        'use_rationales': True,
        'use_rho_conditioning': False,
        'alpha': 0.3  # uniform distillation weight
    },
    'rho_weighting_only': {
        'use_rationales': False,
        'use_rho_conditioning': True,
        'alpha': 0.0  # ρ weights CE, no soft labels
    },
    'rho_conditioned_rationale': {
        'use_rationales': True,
        'use_rho_conditioning': True,
        'alpha': 0.3  # base alpha, ρ-modulated
    }
}
```

#### Hours 5-7: Ablation evaluation

Once training completes (or while Variant 1 trains, run eval on it):

```bash
# Evaluate each variant under Protocol A (editorial pool)
for variant in vanilla rationale_uniform rho_weighting_only rho_conditioned_rationale; do
    python scripts/06_evaluate.py \
        --protocol editorial_pool \
        --reranker cross_encoder \
        --model_variant $variant \
        --output results/ablation_${variant}.json
done

# Evaluate best variant under Protocol B (TREC2023 end-to-end)
python scripts/06_evaluate.py \
    --protocol trec2023 \
    --retriever hybrid \
    --reranker cross_encoder \
    --model_variant rho_conditioned_rationale \
    --output results/trec2023_full_pipeline.json

# Generate the ablation table:
python scripts/18_compile_ablation_table.py
```

**The ablation table (fill in after running):**

| Model | NDCG@10 (editorial pool) | NDCG@10 on low-ρ queries | NDCG@10 on high-ρ queries | Δ vs vanilla |
|---|---|---|---|---|
| BM25 baseline | — | — | — | — |
| ColBERT (existing) | 0.7442 | — | — | — |
| Variant 1: Vanilla CE | — | — | — | baseline |
| Variant 2: Rationale uniform | — | — | — | +X |
| Variant 3: ρ-weighting only | — | — | — | +X |
| Variant 4: ρ-conditioned rationale | — | — | — | +X |

**Key comparison:** Variant 4 vs Variant 2 on low-ρ queries specifically.
If Variant 4 > Variant 2 on low-ρ AND Variant 4 ≈ Variant 2 on high-ρ:
your ρ-conditioning is validated as the contribution.

Run paired Wilcoxon signed-rank test between Variant 4 and Variant 2 NDCG
distributions on low-ρ test queries. Report p-value. Need p < 0.05.

#### Hour 7-8: FINDINGS.md and README update

```markdown
# docs/FINDINGS.md

## Finding 1: Evaluation Bias in ESCI Benchmarks

Every published paper on ESCI (KDD Cup 2022, COLING 2025, TaoSR1, LORE)
evaluates on a pre-selected editorial pool of ~20 candidates per query.
This pool was retrieved by Amazon's production system and cannot be reproduced.

We are the first to evaluate end-to-end retrieval from the full 1.8M catalog,
using TREC Product Search 2023 as the evaluation benchmark — specifically
designed to fix this gap (TREC 2023 paper explicitly states ESCI
"lacks a clear end-to-end retrieval benchmark").

Results:
  BM25 NDCG@10 on editorial pool: X.XXXX
  BM25 NDCG@10 on TREC2023 (full catalog): X.XXXX
  Gap: X.XXXX (editorial pool overstates quality by X%)

This gap quantifies how much existing ESCI benchmarks overstate
search quality relative to what production systems actually achieve.

## Finding 2: ρ-Conditioned Rationale Distillation

[Fill in after ablation results]

## Finding 3: LambdaMART Regression (existing, already documented)

[Already in results/ltr_regression_analysis.json]
```

---

## SECTION 4: WHAT THE FINAL RESULTS TABLE LOOKS LIKE

All numbers under same NDCG formulation: gain=[0,1,3,7].

### Table 1: Re-ranking evaluation (Protocol A — ESCI editorial pool)
*Directly comparable to published ESCI papers*

| System | NDCG@10 | Note |
|---|---|---|
| BM25 baseline | (Day 1 result) | |
| KDD Cup 2022 baseline | 0.8503 | [Reddy et al. 2022] — re-ranking only |
| KDD Cup 2022 winner | 0.9043 | [Zhang et al. 2022] — ensemble, re-ranking only |
| ColBERT (existing, bert-base) | 0.7442 | Your existing result, corrected eval |
| Cross-encoder Variant 1 (vanilla) | (Day 2 result) | |
| Cross-encoder Variant 2 (rationale uniform) | (Day 2 result) | Replicates COLING 2025 |
| Cross-encoder Variant 3 (ρ-weighting only) | (Day 2 result) | |
| Cross-encoder Variant 4 (ρ-conditioned, ours) | (Day 2 result) | **Novel contribution** |

Note clearly: "KDD Cup numbers measure re-ranking on pre-selected pool.
Our numbers measure the same task for direct comparison on Stage 2 only."

### Table 2: End-to-end retrieval evaluation (Protocol B — TREC2023)
*No prior work reports these numbers*

| System | NDCG@10 | Recall@100 (Exact) | Note |
|---|---|---|---|
| BM25 only | (Day 1 result) | — | First honest baseline |
| BM25 + FAISS hybrid | (Day 1 result) | — | |
| Full pipeline (hybrid + cross-encoder V4) | (Day 2 result) | — | |

### Table 3: Ablation on low-ρ queries
*Validates the novel contribution*

| System | NDCG@10 (low-ρ queries) | p-value vs V2 |
|---|---|---|
| Variant 2: Rationale uniform | — | baseline |
| Variant 4: ρ-conditioned rationale | — | — |

---

## SECTION 5: WHAT NOT TO BUILD (FINAL ANSWER)

Do not build any of these. They are removed from scope permanently.

1. **LLM query expansion** — latency non-starter, no production justification
2. **Post-reranking Complement filter** — cross-encoder handles this via training
3. **Tiered serving** — requires production traffic data you don't have
4. **Streamlit demo** — not what impresses ML engineers; costs time better spent elsewhere
5. **Multilingual analysis** — interesting but dilutes focus from the two real contributions
6. **Synonym dictionary for query router** — the query router is pre-retrieval routing only;
   keep it as 3-line rule (token count check), no dictionary needed
7. **Third novel contribution** — two is enough when they're real

---

## SECTION 6: VALIDATION CHECKLIST

Before calling the project done, verify every item:

**Correctness:**
☐ Single NDCG formulation (gain=[0,1,3,7]) used in all scripts
☐ README numbers match results/final_evaluation.json
☐ CI gate runs live evaluation, not static JSON
☐ GCP project name consistent across all scripts
☐ ColBERT special tokens ([Q]/[D]) correctly loaded in serving/app.py
☐ BM25 normalization consistent between train and serve

**Evaluation validity:**
☐ Protocol A (editorial pool) numbers are comparable to published work
☐ Protocol B (TREC2023) uses proper full-catalog qrels
☐ Both protocols clearly labeled in README — never mix them in same table
☐ Coverage reported for Protocol B (what fraction of retrieved products have labels)
☐ Ablation table has all 4 variants with statistical significance test

**Novel contribution validity:**
☐ Variant 4 outperforms Variant 2 on low-ρ queries (p < 0.05)
☐ Variant 4 ≈ Variant 2 on high-ρ queries (ρ-conditioning doesn't hurt easy cases)
☐ FINDINGS.md documents the hypothesis, experiment, result, and interpretation

**What to do if Variant 4 does NOT beat Variant 2:**
This is a valid negative result. Document it honestly:
"We hypothesized ρ-conditioning would improve performance on hard queries.
The improvement was not statistically significant (p=X). Analysis suggests
[reason: not enough low-ρ training examples / rationale quality insufficient /
ρ signal is noisy for this purpose]. The evaluation bias finding (Finding 1)
remains the primary novel contribution."
Do not change the numbers. Report what you found.

---

## SECTION 7: REFERENCES FOR EVERY CLAIM

Every technical claim in README/FINDINGS.md must cite one of these:

- Reddy et al. (2022). ESCI Shopping Queries Dataset. arXiv:2206.06588.
- Zhang et al. (2022). KDD Cup 2022 1st place. arXiv:2208.02958.
- Agrawal et al. (2025). Rationale-Guided Distillation. COLING Industry 2025.
- Dong et al. (2025). TaoSR1. arXiv:2508.12365.
- Lu et al. (2025). LORE. arXiv:2512.03025.
- Xia et al. (2025). From Reasoning LLMs to BERT. arXiv:2510.11056.
- Campos et al. (2023). TREC 2023 Product Search Track. arXiv:2311.07861.
- Warner et al. (2024). ModernBERT. December 2024.
- Cormack et al. (2009). Reciprocal Rank Fusion. SIGIR 2009.
- Khattab & Zaharia (2020). ColBERT. SIGIR 2020. arXiv:2004.12832.
- Burges (2010). LambdaMART. Microsoft Research TR.
- Li et al. (2025). Short Video Relevance Dataset. ByteDance/AAAI 2026. arXiv:2509.16717.

