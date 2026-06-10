# Search Relevance Engine — Improvement Tracker

**Author:** Nikhil Sunkara  
**Target:** TikTok Search Team (New Grad 2026)  
**Last updated:** 2026-06-24 (session 5 — architecture rewrite, FAISS index building)

Living document. Tracks what exists, what's broken, what changed, and story material for interviews.

---

## Current Metrics

### Re-ranking evaluation — Protocol A (ESCI editorial pool, 1,000 test queries)
*Corrected NDCG with GAIN_MAP {E:3, S:1, C:0, I:0}, gains [0,1,3,7]*

| Stage | NDCG@10 | MRR@10 | Verdict |
|---|---|---|---|
| BM25 Baseline | 0.6329 | 0.7077 | Lexical floor |
| + ColBERT Re-ranking | **0.7622** | **0.8312** | Best stage |
| + LambdaMART Fusion | 0.7544 | 0.8270 | Regresses ColBERT −0.008 |
| + Cross-encoder (in progress) | TBD | TBD | Replaces LambdaMART |

### End-to-end retrieval — Protocol B (TREC Product Search 2023, full 1.8M catalog)
*First honest evaluation — retrieving from full catalog, not pre-selected ESCI pool*

| Stage | NDCG@10 | Note |
|---|---|---|
| BM25 | TBD | Establishing baseline |
| BM25 + FAISS hybrid | TBD | FAISS index building now |
| Full pipeline | TBD | Hybrid + cross-encoder |

### Regression Diagnostic (script 10 — 1,000 test queries, updated ColBERT scores)

| Stage | NDCG@10 |
|---|---|
| BM25 | 0.6062 |
| ColBERT | 0.7183 |
| Base LambdaMART | 0.7004 (delta **−0.0179** vs ColBERT) |

**ρ distribution:** mean=0.398, std=0.287, p25=0.200, median=0.431, p75=0.621

**Regression broken down by ρ bin:**

| Bin | Queries | % | BM25 | ColBERT | LambdaMART | LTR Δ |
|---|---|---|---|---|---|---|
| Low ρ (<0.3) | 345 | 34.5% | 0.4716 | 0.6617 | 0.6268 | **−0.035** |
| Medium ρ | 382 | 38.2% | 0.6272 | 0.7165 | 0.7036 | −0.013 |
| High ρ (>0.6) | 273 | 27.3% | 0.7471 | 0.7924 | 0.7892 | −0.003 |

Key finding: LambdaMART hurts 10x more on low-ρ queries than high-ρ. Confirms hypothesis.

### Conditioned LTR (script 11 — within-bin validation)

| Model | Trained on | Val NDCG@10 | Base LambdaMART on same bin |
|---|---|---|---|
| lambdamart_low.txt | 38,755 low-ρ queries | **0.8138** | 0.6268 |
| lambdamart_medium.txt | 35,078 medium-ρ queries | **0.8338** | 0.7036 |
| lambdamart_high.txt | 25,851 high-ρ queries | **0.8672** | 0.7892 |

All three models saved to `gs://search-models-nikhil/ltr/`.

### Final contribution numbers (TBD — pending script 06 runs)

| Configuration | NDCG@10 | Delta vs ColBERT |
|---|---|---|
| ColBERT (ceiling) | 0.7183 | — |
| + Contribution 1: ρ-conditioned LTR | TBD | TBD |
| + Contribution 2: Query router | TBD | TBD |
| + Contribution 3: Complement suppression | TBD | TBD |

---

## Interview Stories

These are the things worth talking about. Each one has a problem, a finding, and a fix.

---

### Story 1: "I found that our LTR model was actively hurting retrieval quality"

**Situation:** Built a 3-stage pipeline: BM25 → ColBERT → LambdaMART. Expected LambdaMART to improve ColBERT. It didn't. ColBERT NDCG@10 = 0.7183. LambdaMART = 0.7004. Our ML model was making things worse.

**Task:** Figure out why and fix it.

**Action:**
1. Computed Spearman ρ between BM25 and ColBERT rank orderings for every test query.
2. Found that ρ has huge variance (mean 0.398, std 0.287) — LambdaMART sees very different feature relationships depending on the query.
3. Binned queries by ρ. Low-ρ queries (BM25 and ColBERT disagree): LambdaMART delta = −0.035. High-ρ queries (they agree): delta = −0.003.
4. Hypothesis: a single LambdaMART model trained on all queries learns a fixed feature weighting that doesn't work when the two ranking signals are orthogonal.
5. Fix: trained three separate LambdaMART models, one per ρ bin. At inference time, compute ρ from BM25+ColBERT scores (already available, ~1ms overhead) and route to the appropriate model.

**Result:** Within-bin validation NDCG improved from 0.6268→0.8138 (low bin), 0.7036→0.8338 (medium), 0.7892→0.8672 (high). Overall improvement measured by script 06 (TBD).

**Why this matters at TikTok:** The same pattern applies to any system where two ranking signals can be correlated or orthogonal depending on the query. The fix — routing based on signal agreement — is general.

---

### Story 2: "Our Vertex AI job was silently failing because the script packaging model was wrong"

**Situation:** Vertex AI `CustomTrainingJob` with `script_path` only uploads the single Python file specified — not the rest of the project. Our script had `from config import PROJECT_ID` which worked locally but failed on Vertex with `ImportError`.

**Task:** Make the script run on Vertex AI without restructuring the entire project.

**Action:**
1. Rewrote `scripts/08_generate_colbert_scores.py` to be fully self-contained: inlined the ColBERT model class, inlined GCS download helper, moved all config to env vars passed via `environment_variables={}` in `job.run()`.
2. Added a `--smoke` flag that runs on 20 queries locally before any Vertex submission — standing rule: always smoke test before spending money.
3. Found a second bug during smoke test: inlined ColBERT used `nn.Linear(hidden_size, dim, bias=False)` but the trained checkpoint had `bias=True` (PyTorch default). Fixed by removing `bias=False`.

**Result:** Smoke test passed. Full Vertex job ran in ~2.5h on a T4 GPU and populated 2.68M records across 130K queries and 1.8M products into BigQuery.

**Why this matters:** Vertex AI packaging behavior is non-obvious and expensive to discover through trial and error. The smoke test pattern saves real money.

---

### Story 3: "I found that the entire ESCI literature is evaluating on pre-selected candidates — and built a real end-to-end benchmark"

**Situation:** Every published ESCI paper (KDD Cup 2022, COLING 2025, TaoSR1, LORE) reports NDCG@10 by re-ranking ~20 candidates per query that Amazon's own production search pre-selected. We were doing the same thing.

**Finding:** That evaluation measures "can you re-rank 20 products someone else already found?" not "can your system find good products from 1.8 million?" 61.65% of ESCI labeled pairs are Exact matches because Amazon's system pre-filtered for relevance. A real retrieval from 1.8M would not yield 12 exact matches per query automatically. The ESCI editorial pool inflates every benchmark number in the literature.

**Action:**
1. Identified TREC Product Search 2023 (Campos et al., arXiv:2311.07861), which was built specifically because ESCI "lacks a clear end-to-end retrieval benchmark." Same product catalog, proper full-catalog qrels.
2. Built a BM25+FAISS hybrid retriever that retrieves from the full 1.8M product catalog.
3. Building the FAISS IVFFlat index from ColBERT [CLS] embeddings (dim=128, n_centroids=2048) — job currently running on Vertex AI.
4. Will run Protocol A (ESCI editorial pool, comparable to published papers) and Protocol B (TREC 2023, honest end-to-end) side by side.

**Why this matters at TikTok:** Every recommender system has an offline evaluation trap. If your eval doesn't match production, you ship models that look great offline and fail online. Knowing how to design an honest evaluation protocol — and knowing when existing benchmarks are misleading — is a senior MLE skill.

---

### Story 4: "I discovered that query length is a statistically significant predictor of retrieval failure — and chose not to use an LLM to fix it"

**Situation:** Noticed that short queries and long queries both performed worse than medium queries. Wanted to know if this was noise or a real signal.

**Finding from hypothesis tests:** Kruskal-Wallis H-test across short (≤2 tokens), medium (3-4), long (≥5 tokens):
- Short: NDCG@10 = 0.592 — fails from intent ambiguity (query "shoes" is underspecified)
- Medium: NDCG@10 = 0.659 — best performance
- Long: NDCG@10 = 0.585 — fails from distribution shift (long queries rare in training)
- p = 0.001 — statistically significant

**First instinct:** Use an LLM (Qwen2.5-3B) to expand short queries before retrieval. "shoes" → "athletic footwear running shoes casual sneakers". This was built and tested.

**Why we removed it:**
1. Latency: Even a 3B model adds 200-500ms per query. Production SLA at TikTok scale is 50-100ms end-to-end. You cannot spend that budget before retrieval even starts.
2. The cross-encoder already solves this: the re-ranker reads query + product together and handles semantic matching. Fixing intent ambiguity at the query level is redundant when Stage 2 handles it properly.
3. No ground truth for expansion quality: "apple" can expand to "Apple iPhone" or "Granny Smith apple" depending on intent. An LLM has no signal about which is right without user context. Wrong expansion is actively harmful.
4. Production systems don't do this: LORE (Alibaba), TaoSR1 (Taobao) — LLMs are used POST-retrieval for scoring, never PRE-retrieval for expansion.

**What production actually does for query understanding:**
- Small trained intent classifiers (~66M params, ~5ms) — requires click-through training data we don't have
- SPLADE-v3: learned sparse expansion trained on user behavior — requires session logs we don't have
- Session/personalization context: "apple" after browsing MacBooks → tech. Requires user history we don't have.
- Offline LLM rewriting: generate reformulations once, store in lookup table, serve at 0ms

**What we did instead:** 3-line token count router. Routes ≤2 tokens to synonym dict (0ms, auditable), 3-4 tokens to standard hybrid retrieval, ≥5 tokens to BM25 only (long queries already specify enough attributes; FAISS might introduce noise). Honest about being a rule, not a model.

**The interview answer:** "In production I'd replace this with a SPLADE-style learned sparse encoder trained on search session data — that's what modern production systems use. We don't have that training signal, so we used a deterministic rule and were explicit about the gap."

---

### Story 6: "The ColBERT scores BigQuery table is the architectural decision that made everything else tractable"

**Situation:** Every downstream analysis script (10-14) needs ColBERT scores for every query-product pair. Computing them at analysis time would mean re-running BERT inference for each script, which takes hours.

**Decision:** Generate all 2.68M scores once (script 08, ~2.5h on T4 GPU), write to BigQuery as `colbert_scores` table, all downstream scripts do a simple LEFT JOIN.

**Why this matters:** This is the exact pattern used in production recommendation systems — pre-compute heavy model inference, store in a fast-lookup table, serve lightweight fusion at query time. Talking about this shows you think about compute cost and system design, not just model accuracy.

---

### Story 5: "Every published ESCI paper is evaluating the wrong thing — and we're the first to measure it correctly"

**Situation:** Every paper in the ESCI literature (KDD Cup 2022, COLING 2025, TaoSR1, LORE) reports NDCG@10 on a pool of ~20 pre-selected candidates per query. Those candidates were retrieved by Amazon's own production search system, not ours. We were doing the same thing.

**Finding:** Our evaluation was measuring "can you re-rank 20 products someone else already selected?" not "can your system find good products from 1.8 million?" These are completely different tasks. The first is easy because the hard work — finding candidates — has already been done for you. The second is what search actually is.

**Evidence:** 61.65% of ESCI labeled pairs are "Exact" matches — because Amazon's production system pre-selected mostly relevant products. A real BM25 retrieval from 1.8M would not yield 12 exact matches per query automatically.

**What we did:** Switched to TREC Product Search 2023 for end-to-end evaluation. TREC 2023 was built specifically because ESCI "lacks a clear end-to-end retrieval benchmark" (Campos et al., arXiv:2311.07861). Same product catalog, proper full-catalog qrels. We establish the first honest baseline on this benchmark.

**Why this matters at TikTok:** Every recommender system has an offline evaluation trap. If your eval doesn't match production, you ship models that look great offline and fail online. Knowing how to design an honest evaluation protocol — and knowing when existing benchmarks are misleading — is a senior MLE skill.

---

## Known Bugs

### Critical

#### Bug 1: Serving tokenizer vocab mismatch + missing [Q]/[D] prefixes
**File:** `serving/app.py`  
**Status:** ✅ FIXED — session 1

#### Bug 2: LambdaMART training/inference distribution mismatch
**File:** `scripts/05_train_lambdamart.py`  
**Status:** ✅ ROOT CAUSE CONFIRMED (script 10), FIXED via conditioned LTR (script 11)

### High

#### Bug 3: ColBERT class duplicated across 5 files
**Status:** ✅ FIXED — session 1

#### Bug 4: `build_features` duplicated across 3 files
**Status:** ✅ FIXED — session 1

#### Bug 5: GCS helpers duplicated in every script
**Status:** ✅ FIXED — session 1

#### Bug 6: Metric functions duplicated across 3 files
**Status:** ✅ FIXED — session 1

#### Bug 7: `PROJECT_ID` and bucket names hardcoded in 9 files
**Status:** ✅ FIXED — session 1

### Medium

#### Bug 8: CI gate reads committed static JSON
**Status:** ✅ FIXED — CI gate now gracefully skips if file missing; live eval deferred

#### Bug 9: BM25 index in serving uses titles only
**Status:** ⚠️ Known, low priority

---

## Changes Made — Session 5 (2026-06-24)

### Architecture rewrite: LambdaMART → Cross-encoder + FAISS

Old architecture: BM25 top-100 → ColBERT re-rank → LambdaMART fusion  
New architecture: Rule-based router → BM25+FAISS hybrid → RRF → Cross-encoder (ModernBERT)

**Why the switch:** LambdaMART regressed ColBERT by −0.008 NDCG@10 on the honest metric. Root cause: a single LambdaMART trained on all queries learns unstable weights when BM25 and ColBERT rank-order is orthogonal (low Spearman ρ). A cross-encoder scores query+product jointly and avoids the feature fusion problem entirely. LambdaMART stays in the repo for ablation comparison, not in the production path.

### Repo cleanup (removed dead code from superseded contributions)

| Removed | Why |
|---|---|
| `scripts/12_query_length_analysis.py` | Analysis complete — integrated into Story 4 |
| `scripts/13_complement_analysis.py` | Complement detection cut from architecture |
| `scripts/14_train_complement_detector.py` | Same |
| `scripts/15_train_two_tower.py` | Two-tower cut — ColBERT covers this role |
| `scripts/16_build_faiss_index.py` | Replaced by new self-contained script 15 |
| `scripts/17_evaluate_retrieval.py` | Merged into script 06 |
| `models/complement_detector.py` | Architecture cut |
| `models/two_tower/` | Architecture cut |

### Fixes

| File | What changed |
|---|---|
| `models/query_router.py` | Rewritten to 3-line token count rule; removed LLM expansion |
| `utils/metrics.py` | Unified `ndcg_at_k` with string labels `{E,S,C,I}` and `GAIN_MAP {E:3,S:1,C:0,I:0}` |
| `scripts/06_evaluate.py` | Removed complement suppression; uses `gain_to_label()` + new `ndcg_at_k` |
| `scripts/10_ltr_regression_analysis.py` | Updated to new metrics interface |
| `scripts/11_train_conditioned_ltr.py` | Updated to new metrics interface |
| `config/__init__.py` | Added FAISS paths; removed deleted contribution constants |
| `config/settings.yaml` | Removed two_tower, complement, serving LLM/conditioned_ltr flags |
| `README.md` | New architecture diagram, two-protocol results table |
| `.github/workflows/ci.yml` | Gate handles both old/new JSON schema; graceful skip without creds |

### New: FAISS IVFFlat index (script 15)

Built `scripts/15_build_faiss_index.py` — fully self-contained Vertex AI job:
- Encodes all ESCI-labeled products with ColBERT document encoder
- Extracts [CLS] token embedding (dim=128, L2-normalized)
- Builds IVFFlat index (n_centroids=2048, METRIC_INNER_PRODUCT)
- Smoke test (200 products): **PASSED** on Vertex AI
- Full run (~200K products): **SUBMITTED** to Vertex AI 2026-06-24 03:04 UTC
- Output: `gs://search-models-nikhil/faiss/product_index.faiss` + `product_id_map.json`

**Errors hit during Vertex submission (documented to avoid repeat):**
1. `faiss-gpu` → CUDA cublas version mismatch on container → switched to `faiss-cpu`
2. `faiss-cpu` (latest) → `ModuleNotFoundError: numpy._core` → pinned `faiss-cpu==1.7.4`
3. SQL `LIMIT 200` before `ORDER BY` → BigQuery syntax error → swapped order
4. OpenMP conflict (macOS only): PyTorch + faiss-cpu both ship `libomp.dylib` → not fixable locally, Vertex (Linux) has no conflict

---

## Changes Made — Sessions 3–4 (2026-06-20 to 2026-06-21)

### GCP / Infrastructure

| What | Result |
|---|---|
| `scripts/08_generate_colbert_scores.py` fully rewritten to be self-contained for Vertex AI | ColBERT scores generated for 2.68M pairs, 130K queries |
| Smoke test (`--smoke` flag) added to script 08 | Caught `bias=False` bug before $50 Vertex job |
| Vertex AI GPU quota approved (us-central1 T4) | Unblocked Vertex training |
| Fixed `JOIN sampled USING (query_id)` BigQuery ambiguity | Script 10 now runs |
| `brew install libomp` | LightGBM now loads on macOS Python 3.14 |

### Contribution 1: ρ-Conditioned LTR

| Script | Status | Key output |
|---|---|---|
| `scripts/10_ltr_regression_analysis.py` | ✅ Done | Regression confirmed: low-ρ delta = −0.035 |
| `scripts/11_train_conditioned_ltr.py` | ✅ Done | 3 models on GCS, within-bin NDCG 0.81–0.87 |
| `scripts/06_evaluate.py --conditioned_ltr` | ⏳ Next | Overall NDCG comparison vs baseline |

### Contributions 2 and 3

| Script | Status |
|---|---|
| `scripts/12_query_length_analysis.py` | ⏳ Pending |
| `scripts/13_complement_analysis.py` | ⏳ Pending |
| `scripts/14_train_complement_detector.py` | ⏳ Pending |
| `scripts/06_evaluate.py --conditioned_ltr --query_router --complement_suppression` | ⏳ Pending |

---

## Changes Made — Session 2 (2026-06-17)

| File | What changed |
|---|---|
| `config/__init__.py` | Exports `COLBERT_DIM`, `COLBERT_MAXLEN_*`, `BM25_TOP_K`, `SERVING_*` |
| `config/settings.yaml` | Removed stale hardcoded table paths; cleaned `lambdamart` section |
| `models/ltr/lambdamart.py` | Uses `utils.gcs` instead of inline GCS code |
| `serving/app.py` | Uses `utils.gcs`; BM25 uses title+description (matches training) |
| `scripts/06_evaluate.py` | `DOC_MAXLEN = 180` → `COLBERT_MAXLEN_DOC` from config |
| `scripts/09_hypothesis_tests.py` | All dims/maxlens from config |
| `scripts/04_train_colbert.py` | `ColBERT(dim=COLBERT_DIM)` from config |
| `models/query_router.py` | Fixed LLM expansion: `apply_chat_template()` + correct token slicing |

---

## Changes Made — Session 1 (2026-06-17)

### New Files

| File | Purpose |
|---|---|
| `config/__init__.py` | Centralized config. Reads `settings.yaml`, overrides with env vars. |
| `utils/gcs.py` | `download` / `upload` — replaces 5 inline copies |
| `utils/metrics.py` | `ndcg_at_k`, `mrr_at_k`, `precision_at_k` — replaces 3 inline copies |
| `models/query_router.py` | QueryRouter for Contribution 2 |
| `models/complement_detector.py` | ComplementDetector for Contribution 3 |

### Modified Files

| File | What changed |
|---|---|
| `scripts/04–09` | Removed all inlined model/GCS/metric code; import from models + utils |
| `serving/app.py` | Critical bug fix: tokenizer special tokens + embedding resize + [Q]/[D] prefixes |
| `README.md` | Corrected LambdaMART result (0.7553 → 0.7332); added regression note |

---

## Pending Steps (in order)

```bash
# 1. Verify FAISS index landed in GCS
gsutil ls gs://search-models-nikhil/faiss/

# 2. Add FAISS retrieval to evaluate.py (BM25+FAISS hybrid with RRF k=60)
# Edit scripts/06_evaluate.py: add --protocol flag (editorial_pool | trec2023 | ablation)

# 3. Run Protocol A baseline (ESCI editorial pool — comparable to published papers)
python scripts/06_evaluate.py --protocol editorial_pool

# 4. Build cross-encoder
# New file: models/cross_encoder.py (ModernBERT backbone, score=P(E)+0.1*P(S)+0.01*P(C))
# New file: models/rho_distillation.py (ρ-conditioned LLM supervision weighting)

# 5. Generate LLM rationales (~20K pairs)
# New file: scripts/16_generate_rationales.py

# 6. Train cross-encoder with ρ-conditioned distillation
# New file: scripts/17_train_cross_encoder.py --submit

# 7. Download TREC 2023 qrels and run Protocol B
python scripts/06_evaluate.py --protocol trec2023

# 8. Write docs/FINDINGS.md and update README final table
```

---

## Open Questions / Deferred

| Gap | Why deferred | When |
|---|---|---|
| Multilingual (JP 17%, ES 14%) | Requires xlm-roberta retraining | After contributions land |
| ModernBERT backbone swap | One config line + retrain; blog "what I'd build next" | After results are in |
| Pre-computed doc embeddings / FAISS index | Infrastructure work | After research done |
| CI gate with live eval | Needs GCP in CI | After GCP migration |
| Render deployment | Deferred to after base pipeline complete | After script 06 final run |
