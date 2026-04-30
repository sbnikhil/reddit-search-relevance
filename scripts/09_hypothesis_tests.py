"""
Formal statistical hypothesis testing for the 3-stage search pipeline.

Tests
-----
1. Paired t-test: ColBERT vs BM25 NDCG@10 (per query)
2. Paired t-test: LambdaMART vs ColBERT NDCG@10 (per query)
3. Kruskal-Wallis + post-hoc Mann-Whitney U:
   BM25 score distributions across ESCI relevance labels
"""

import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from google.cloud import bigquery, storage
from rank_bm25 import BM25Okapi
from scipy import stats
from transformers import BertModel, BertTokenizerFast

PROJECT_ID    = "reddit-search-relevance-485717"
MODELS_BUCKET = "reddit-search-relevance-models"
COLBERT_CKPT  = f"gs://{MODELS_BUCKET}/colbert/epoch_5/model.pt"
LTR_MODEL_GCS = f"gs://{MODELS_BUCKET}/ltr/lambdamart.txt"
COLBERT_DIM   = 128
QUERY_MAXLEN  = 32
DOC_MAXLEN    = 180
DOC_BATCH     = 64
SAMPLE_QUERIES = 1000
RESULTS_DIR   = "results"
ALPHA         = 0.05

FEATURE_NAMES = [
    "bm25_score", "colbert_score", "title_query_overlap",
    "desc_query_overlap", "title_bigram_overlap", "brand_match",
    "title_length", "desc_length", "has_description", "query_length",
]


class ColBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dim=128):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim)

    def encode_query(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(self.linear(out.last_hidden_state), p=2, dim=-1)

    def encode_document(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embs = F.normalize(self.linear(out.last_hidden_state), p=2, dim=-1)
        return embs * attention_mask.unsqueeze(-1).float()


def _download(gcs_uri: str, local_path: str) -> None:
    bucket, blob = gcs_uri.replace("gs://", "").split("/", 1)
    storage.Client().bucket(bucket).blob(blob).download_to_filename(local_path)


def _dcg(gains: list, k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(pred: list, ideal: list, k: int) -> float:
    idcg = _dcg(sorted(ideal, reverse=True), k)
    return _dcg(pred, k) / idcg if idcg > 0 else 0.0


def _bigrams(tokens):
    return set(zip(tokens, tokens[1:]))


def build_features(query, title, description, brand, bm25_score=0.0, colbert_score=0.0):
    q_tok = query.lower().split()
    t_tok = (title or "").lower().split()
    d_tok = (description or "").lower().split()
    brand = (brand or "").lower()
    q_set = set(q_tok)
    q_bg, t_bg = _bigrams(q_tok), _bigrams(t_tok)
    return [
        bm25_score,
        colbert_score,
        len(q_set & set(t_tok)) / max(len(q_set), 1),
        len(q_set & set(d_tok)) / max(len(q_set), 1),
        len(q_bg & t_bg) / max(len(q_bg), 1) if q_bg else 0.0,
        float(bool(brand) and brand in query.lower()),
        np.log1p(len(t_tok)),
        np.log1p(len(d_tok)),
        float(bool(d_tok)),
        float(len(q_tok)),
    ]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return diff.mean() / diff.std(ddof=1)


def load_test_data() -> pd.DataFrame:
    bq = bigquery.Client(project=PROJECT_ID)
    sql = f"""
        WITH sampled AS (
            SELECT DISTINCT query_id
            FROM `{PROJECT_ID}.esci_search.examples`
            WHERE split = 'test'
            LIMIT {SAMPLE_QUERIES}
        )
        SELECT e.query_id, e.query, e.product_id, e.gain, e.esci_label,
               COALESCE(p.product_title, '')       AS product_title,
               COALESCE(p.product_description, '') AS product_description,
               COALESCE(p.product_brand, '')        AS product_brand
        FROM `{PROJECT_ID}.esci_search.examples` e
        JOIN `{PROJECT_ID}.esci_search.products` p USING (product_id)
        JOIN sampled USING (query_id)
    """
    print(f"  Loading {SAMPLE_QUERIES} test queries from BigQuery...")
    df = bq.query(sql).to_dataframe()
    print(f"  {len(df):,} query-product pairs loaded.")
    return df


def compute_per_query_ndcg(df: pd.DataFrame, score_col: str) -> np.ndarray:
    scores = []
    for _, group in df.groupby("query_id"):
        ranked = group.sort_values(score_col, ascending=False)
        scores.append(ndcg_at_k(ranked["gain"].tolist(), group["gain"].tolist(), 10))
    return np.array(scores)


def add_bm25_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query_id, group in df.groupby("query_id"):
        query  = group["query"].iloc[0]
        corpus = (group["product_title"].fillna("") + " " + group["product_description"].fillna("")).tolist()
        raw    = BM25Okapi([t.lower().split() for t in corpus]).get_scores(query.lower().split())
        norm   = raw / (raw.max() or 1.0)
        for pid, s in zip(group["product_id"], norm):
            rows.append({"product_id": pid, "query_id": query_id, "bm25_score": s})
    return df.merge(pd.DataFrame(rows), on=["query_id", "product_id"])


def add_colbert_scores(df: pd.DataFrame, model: ColBERT, tokenizer, device) -> pd.DataFrame:
    rows = []
    for query_id, group in df.groupby("query_id"):
        query_text = group["query"].iloc[0]
        titles     = group["product_title"].fillna("").tolist()
        pids       = group["product_id"].tolist()

        q_enc = tokenizer([f"[Q] {query_text}"], padding=True, truncation=True,
                          max_length=QUERY_MAXLEN, return_tensors="pt")
        with torch.no_grad():
            q_emb = model.encode_query(
                q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device)
            )[0]

        scores = []
        for start in range(0, len(titles), DOC_BATCH):
            batch = [f"[D] {t}" for t in titles[start : start + DOC_BATCH]]
            d_enc = tokenizer(batch, padding=True, truncation=True,
                              max_length=DOC_MAXLEN, return_tensors="pt")
            with torch.no_grad():
                d_embs = model.encode_document(
                    d_enc["input_ids"].to(device), d_enc["attention_mask"].to(device)
                )
            sim = torch.einsum("qd,bld->bql", q_emb, d_embs)
            scores.extend(sim.max(dim=2).values.sum(dim=1).cpu().numpy().tolist())

        for pid, s in zip(pids, scores):
            rows.append({"product_id": pid, "query_id": query_id, "colbert_score": s})
    return df.merge(pd.DataFrame(rows), on=["query_id", "product_id"])


def add_ltr_scores(df: pd.DataFrame, booster: lgb.Booster) -> pd.DataFrame:
    features = np.array([
        build_features(
            row["query"], row["product_title"], row["product_description"], row["product_brand"],
            bm25_score=row["bm25_score"], colbert_score=row["colbert_score"],
        )
        for _, row in df.iterrows()
    ], dtype=np.float32)
    df = df.copy()
    df["ltr_score"] = booster.predict(features)
    return df


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_ttest_result(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    t_stat, p_value = stats.ttest_rel(a, b)
    d = cohens_d(a, b)
    sig = p_value < ALPHA

    print(f"\n  H0: mean NDCG@10({name.split(' vs ')[0]}) = mean NDCG@10({name.split(' vs ')[1]})")
    print(f"  H1: {name.split(' vs ')[0]} > {name.split(' vs ')[1]}")
    print(f"\n  Mean NDCG@10  [{name.split(' vs ')[0]}]: {a.mean():.4f} ± {a.std():.4f}")
    print(f"  Mean NDCG@10  [{name.split(' vs ')[1]}]: {b.mean():.4f} ± {b.std():.4f}")
    print(f"  Mean delta:    {(a - b).mean():+.4f}")
    print(f"\n  t-statistic:   {t_stat:.4f}")
    print(f"  p-value:       {p_value:.2e}")
    print(f"  Cohen's d:     {d:.4f}  ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'} effect)")
    print(f"  alpha:         {ALPHA}")
    print(f"\n  {'REJECT H0' if sig else 'FAIL TO REJECT H0'} — improvement is {'statistically significant' if sig else 'NOT statistically significant'} (p {'<' if sig else '>='} {ALPHA})")

    return {"t_stat": t_stat, "p_value": p_value, "cohens_d": d, "significant": sig,
            "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def print_kruskal_result(df: pd.DataFrame) -> dict:
    label_order = ["E", "S", "C", "I"]
    label_names = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}
    groups = {l: df[df["esci_label"] == l]["bm25_score"].values for l in label_order}

    print(f"\n  H0: BM25 score distribution is identical across all relevance labels")
    print(f"  H1: At least one label has a different BM25 score distribution")
    print()
    for l in label_order:
        g = groups[l]
        print(f"  {label_names[l]:<14} n={len(g):>6,}   median={np.median(g):.4f}   mean={g.mean():.4f}")

    stat, p_value = stats.kruskal(*[groups[l] for l in label_order])
    n = len(df)
    eta_sq = (stat - len(label_order) + 1) / (n - len(label_order))
    sig = p_value < ALPHA

    print(f"\n  Kruskal-Wallis H:  {stat:.4f}")
    print(f"  p-value:           {p_value:.2e}")
    print(f"  Eta-squared:       {eta_sq:.4f}  ({'large' if eta_sq > 0.14 else 'medium' if eta_sq > 0.06 else 'small'} effect)")
    print(f"\n  {'REJECT H0' if sig else 'FAIL TO REJECT H0'} — BM25 scores differ {'significantly' if sig else 'NOT significantly'} across labels (p {'<' if sig else '>='} {ALPHA})")

    print(f"\n  Post-hoc pairwise Mann-Whitney U (Bonferroni-corrected alpha = {ALPHA / 6:.4f}):")
    pairs = [(l1, l2) for i, l1 in enumerate(label_order) for l2 in label_order[i+1:]]
    pairwise = {}
    for l1, l2 in pairs:
        u_stat, p = stats.mannwhitneyu(groups[l1], groups[l2], alternative="two-sided")
        p_corr = min(p * 6, 1.0)
        sig_pair = p_corr < ALPHA
        pairwise[f"{l1}_vs_{l2}"] = {"u_stat": u_stat, "p_corrected": p_corr, "significant": sig_pair}
        marker = "*" if sig_pair else " "
        print(f"  {marker} {label_names[l1]:<14} vs {label_names[l2]:<14}  U={u_stat:.0f}   p={p_corr:.2e}  {'significant' if sig_pair else 'ns'}")

    return {"kruskal_h": stat, "p_value": p_value, "eta_squared": eta_sq,
            "significant": sig, "pairwise": pairwise}


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print_section("Loading Data")
    df = load_test_data()

    print_section("Stage 1 — BM25 Scoring")
    print("  Computing BM25 scores...")
    df = add_bm25_scores(df)

    print_section("Stage 2 — ColBERT Scoring")
    print("  Loading ColBERT checkpoint...")
    _download(COLBERT_CKPT, "/tmp/colbert_hyp.pt")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})
    colbert = ColBERT(dim=COLBERT_DIM)
    colbert.bert.resize_token_embeddings(len(tokenizer))
    colbert.load_state_dict(torch.load("/tmp/colbert_hyp.pt", map_location=device))
    colbert.to(device).eval()
    print(f"  ColBERT loaded ({device}). Running inference on {SAMPLE_QUERIES} queries...")
    df = add_colbert_scores(df, colbert, tokenizer, device)

    print_section("Stage 3 — LambdaMART Scoring")
    print("  Loading LambdaMART model...")
    _download(LTR_MODEL_GCS, "/tmp/lambdamart_hyp.txt")
    booster = lgb.Booster(model_file="/tmp/lambdamart_hyp.txt")
    df = add_ltr_scores(df, booster)
    print("  Done.")

    bm25_ndcg   = compute_per_query_ndcg(df, "bm25_score")
    cb_ndcg     = compute_per_query_ndcg(df, "colbert_score")
    ltr_ndcg    = compute_per_query_ndcg(df, "ltr_score")

    results = {}

    print_section("Hypothesis Test 1 — ColBERT vs BM25 (Paired t-test, n=1000 queries)")
    results["test1_colbert_vs_bm25"] = print_ttest_result("ColBERT vs BM25", cb_ndcg, bm25_ndcg)

    print_section("Hypothesis Test 2 — LambdaMART vs ColBERT (Paired t-test, n=1000 queries)")
    results["test2_ltr_vs_colbert"] = print_ttest_result("LambdaMART vs ColBERT", ltr_ndcg, cb_ndcg)

    print_section("Hypothesis Test 3 — BM25 Score Distribution by Relevance Label (Kruskal-Wallis)")
    results["test3_bm25_by_label"] = print_kruskal_result(df)

    print_section("Summary")
    print(f"\n  {'Test':<45} {'Result'}")
    print(f"  {'-'*60}")
    t1 = results["test1_colbert_vs_bm25"]
    t2 = results["test2_ltr_vs_colbert"]
    t3 = results["test3_bm25_by_label"]
    print(f"  {'ColBERT > BM25 (NDCG@10)':<45} {'SIGNIFICANT' if t1['significant'] else 'not significant':>12}   p={t1['p_value']:.2e}")
    print(f"  {'LambdaMART > ColBERT (NDCG@10)':<45} {'SIGNIFICANT' if t2['significant'] else 'not significant':>12}   p={t2['p_value']:.2e}")
    print(f"  {'BM25 scores differ by label':<45} {'SIGNIFICANT' if t3['significant'] else 'not significant':>12}   p={t3['p_value']:.2e}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "hypothesis_tests.json")

    def _json_safe(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    print(f"\n  Full results saved -> {out_path}")


if __name__ == "__main__":
    main()
