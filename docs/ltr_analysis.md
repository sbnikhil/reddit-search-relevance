# LambdaMART Analysis: What Went Wrong and What's Next

## Current Results

Evaluated on 1,000 randomly sampled test queries from the ESCI dataset.

| Stage | NDCG@10 | MRR@10 | P@10 |
|---|---|---|---|
| BM25 Baseline | 0.6274 | 0.7170 | 0.9195 |
| + ColBERT Re-ranking | **0.7442** | **0.8216** | 0.9390 |
| + LambdaMART Fusion | 0.7332 | 0.8046 | 0.9370 |

ColBERT re-ranking delivers a **+18.6% NDCG@10 improvement** over BM25. LambdaMART, however, scores *below* ColBERT alone — which means the LTR stage is hurting rather than helping.

---

## Why LambdaMART Underperforms

LambdaMART is a re-ranker. Its job is to take a candidate list produced by an upstream retriever and reorder it better. The problem is a mismatch between what it was trained on and what it sees at inference time.

### Training distribution

The ESCI dataset provides ~20 hand-picked, Amazon-annotated candidates per query. We trained LambdaMART to re-rank those 20 products. BM25 and ColBERT features were computed within this small, editorially-selected candidate pool.

### Inference distribution

At inference time, BM25 retrieves the top-100 candidates from the full product catalog (~900K products). LambdaMART is then asked to re-rank these 100 BM25 results — a completely different candidate set with very different feature distributions.

### The consequence

LambdaMART learned to rank Amazon's editorial candidates. At inference it sees BM25 retrievals. The feature ranges, score distributions, and candidate quality profiles are all different. The model has no frame of reference for the task it's actually being asked to do.

A secondary issue: ColBERT scores were stored as raw unnormalized MaxSim values during training (range ~[0, 32]), but were mistakenly normalized to [0, 1] during the first evaluation run. This caused LambdaMART's learned thresholds to misfire entirely, producing NDCG@10 = 0.6107. That bug has since been fixed.

---

## The Fix: Train on BM25 Candidates

The right approach is to generate training data that matches inference exactly.

**For each training query:**
1. Run BM25 on the full product catalog → retrieve top-30 candidates
2. Score those 30 candidates with the trained ColBERT checkpoint (inference only, no retraining)
3. Look up ESCI relevance labels for any candidates that appear in the dataset; treat unlabeled candidates as relevance 0
4. Train LambdaMART to re-rank this BM25-retrieved list

Now training and inference see the same kind of candidate list — BM25 retrievals from the full catalog. LambdaMART learns the actual task: *given what BM25 produces, how do I improve the ordering?*

### Why top-30 and not top-100

The ESCI dataset has an average of ~20 judged candidates per query. Using top-100 BM25 candidates means ~80 of them will have no relevance label and be treated as irrelevant (label 0). This is noisy but acceptable. Using top-30 keeps the candidate set closer to the ESCI judgment pool, maximizes label coverage, and reduces compute cost (~3M pairs vs ~13M for top-100).

### Expected outcome

With training and inference aligned, LambdaMART should be able to combine ColBERT's semantic signal with BM25's exact-match signal and text overlap features to push NDCG@10 above 0.7442 (ColBERT alone). The feature importance from the current model already confirms ColBERT is the dominant signal — the question is whether LambdaMART can use the other features to correct ColBERT's mistakes on edge cases (short queries, brand matches, exact title matches).

---

## What Needs to Be Done

| Step | What | Estimated Time |
|---|---|---|
| New step 08 | BM25 top-30 candidate scoring with ColBERT | ~2.5h on Vertex AI T4 |
| Step 05 | Retrain LambdaMART on BM25 candidates | ~20 min locally |
| Step 06 | Re-evaluate full pipeline | ~20 min locally |

ColBERT model training does **not** need to be repeated. The saved checkpoint is reused for inference only.

---

## Feature Importance (Current Model)

Trained on ESCI pairs. Shows ColBERT dominance.

| Feature | Relative Importance |
|---|---|
| colbert_score | ██████████████████████████████ 433,034 |
| title_query_overlap | ██████████████ 214,759 |
| bm25_score | ███████ 105,910 |
| title_length | ██ 33,594 |
| query_length | █ 19,889 |
| brand_match | █ 15,856 |
| title_bigram_overlap | 13,326 |
| desc_length | 4,848 |
| desc_query_overlap | 4,357 |
| has_description | 111 |
