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
COLBERT_DIM   = 128
QUERY_MAXLEN  = 32
DOC_MAXLEN    = 180
DOC_BATCH     = 64
SAMPLE_QUERIES = 1000
RESULTS_DIR   = "results"
ALPHA         = 0.05


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


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.mean() / diff.std(ddof=1))


def print_section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


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


def compute_per_query_ndcg(df: pd.DataFrame, score_col: str, k: int = 10) -> np.ndarray:
    scores = []
    for _, group in df.groupby("query_id"):
        ranked = group.sort_values(score_col, ascending=False)
        scores.append(ndcg_at_k(ranked["gain"].tolist(), group["gain"].tolist(), k))
    return np.array(scores)


def compute_per_query_rank_correlation(df: pd.DataFrame) -> np.ndarray:
    corrs = []
    for _, group in df.groupby("query_id"):
        if len(group) < 3:
            continue
        if group["bm25_score"].nunique() < 2 or group["colbert_score"].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(group["bm25_score"], group["colbert_score"])
        if not np.isnan(rho):
            corrs.append(rho)
    return np.array(corrs)


def _json_safe(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def run_test1_normality(bm25_ndcg: np.ndarray, cb_ndcg: np.ndarray) -> dict:
    print_section("Test 1 — Shapiro-Wilk Normality (BM25 and ColBERT NDCG@10)")
    results = {}
    for name, arr in [("BM25", bm25_ndcg), ("ColBERT", cb_ndcg)]:
        stat, p = stats.shapiro(arr[:500])  # Shapiro-Wilk capped at n=5000; 500 is sufficient
        normal = p >= ALPHA
        results[name] = {"statistic": float(stat), "p_value": float(p), "normal": bool(normal)}
        print(f"\n  {name} NDCG@10 (n={len(arr)})")
        print(f"    W = {stat:.4f},  p = {p:.4e}")
        print(f"    {'Fail to reject' if normal else 'REJECT'} H0 — distribution is {'approximately normal' if normal else 'NOT normal'}")

    bm25_normal = results["BM25"]["normal"]
    cb_normal   = results["ColBERT"]["normal"]
    print(f"\n  => Both normal: {bm25_normal and cb_normal}. "
          f"{'Paired t-test is appropriate.' if bm25_normal and cb_normal else 'Use Wilcoxon signed-rank (non-parametric) as primary test.'}")
    return results


def run_test2_colbert_vs_bm25(bm25_ndcg: np.ndarray, cb_ndcg: np.ndarray) -> dict:
    print_section("Test 2 — ColBERT vs BM25: Paired t-test + Wilcoxon Signed-Rank (n=1000 queries)")
    print(f"\n  H0: ColBERT and BM25 produce equal NDCG@10")
    print(f"  H1: ColBERT NDCG@10 > BM25 NDCG@10")

    t_stat, t_p   = stats.ttest_rel(cb_ndcg, bm25_ndcg, alternative="greater")
    w_stat, w_p   = stats.wilcoxon(cb_ndcg, bm25_ndcg, alternative="greater")
    d             = cohens_d(cb_ndcg, bm25_ndcg)

    print(f"\n  Mean NDCG@10  [ColBERT]: {cb_ndcg.mean():.4f} ± {cb_ndcg.std():.4f}")
    print(f"  Mean NDCG@10  [BM25]:    {bm25_ndcg.mean():.4f} ± {bm25_ndcg.std():.4f}")
    print(f"  Mean delta:              {(cb_ndcg - bm25_ndcg).mean():+.4f}")
    print(f"\n  Paired t-test:     t = {t_stat:.4f},  p = {t_p:.2e}  {'*' if t_p < ALPHA else ''}")
    print(f"  Wilcoxon W-test:   W = {w_stat:.0f},   p = {w_p:.2e}  {'*' if w_p < ALPHA else ''}")
    print(f"  Cohen's d:         {d:.4f}  ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'} effect)")
    print(f"\n  Both tests {'REJECT H0' if t_p < ALPHA and w_p < ALPHA else 'FAIL TO REJECT H0'} — "
          f"ColBERT improvement is statistically significant.")

    return {
        "mean_colbert": float(cb_ndcg.mean()), "mean_bm25": float(bm25_ndcg.mean()),
        "mean_delta": float((cb_ndcg - bm25_ndcg).mean()),
        "ttest": {"t_stat": float(t_stat), "p_value": float(t_p), "significant": bool(t_p < ALPHA)},
        "wilcoxon": {"w_stat": float(w_stat), "p_value": float(w_p), "significant": bool(w_p < ALPHA)},
        "cohens_d": d,
    }


def run_test3_ndcg_cutoffs(df: pd.DataFrame) -> dict:
    print_section("Test 3 — ColBERT vs BM25 Across NDCG Cutoffs (@1, @5, @10)")
    print(f"\n  H0: No difference at each cutoff. H1: ColBERT > BM25 at each cutoff.")
    results = {}
    print(f"\n  {'Cutoff':<10} {'BM25':>8} {'ColBERT':>10} {'Delta':>8} {'p-value':>12} {'Sig':>5}")
    print(f"  {'-' * 55}")
    for k in [1, 5, 10]:
        bm25_k = compute_per_query_ndcg(df, "bm25_score", k)
        cb_k   = compute_per_query_ndcg(df, "colbert_score", k)
        _, p   = stats.wilcoxon(cb_k, bm25_k, alternative="greater")
        sig    = p < ALPHA
        results[f"ndcg_{k}"] = {
            "mean_bm25": float(bm25_k.mean()), "mean_colbert": float(cb_k.mean()),
            "delta": float((cb_k - bm25_k).mean()), "p_value": float(p), "significant": bool(sig),
        }
        print(f"  NDCG@{k:<5}  {bm25_k.mean():>8.4f}  {cb_k.mean():>10.4f}  {(cb_k-bm25_k).mean():>+8.4f}  {p:>12.2e}  {'*' if sig else ' ':>5}")
    return results


def run_test4_spearman(df: pd.DataFrame) -> dict:
    print_section("Test 4 — Spearman Rank Correlation: BM25 vs ColBERT (per query)")
    corrs = compute_per_query_rank_correlation(df)
    t_stat, p = stats.ttest_1samp(corrs, popmean=0)

    print(f"\n  H0: Mean Spearman rho = 0 (BM25 and ColBERT rankings are uncorrelated)")
    print(f"  H1: Mean Spearman rho != 0")
    print(f"\n  Mean rho:    {corrs.mean():.4f} ± {corrs.std():.4f}")
    print(f"  Median rho:  {np.median(corrs):.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p:.2e}")
    print(f"\n  {'REJECT H0' if p < ALPHA else 'FAIL TO REJECT H0'} — rankings are {'significantly correlated' if p < ALPHA else 'not significantly correlated'}.")
    print(f"\n  Interpretation: rho = {corrs.mean():.4f} indicates "
          f"{'moderate' if 0.4 <= abs(corrs.mean()) < 0.7 else 'weak' if abs(corrs.mean()) < 0.4 else 'strong'} "
          f"agreement between BM25 and ColBERT rankings — ColBERT captures "
          f"{'substantial independent signal' if corrs.mean() < 0.5 else 'mostly similar signal to BM25'}.")

    return {
        "mean_rho": float(corrs.mean()), "median_rho": float(np.median(corrs)),
        "std_rho": float(corrs.std()), "t_stat": float(t_stat),
        "p_value": float(p), "significant": bool(p < ALPHA),
    }


def run_test5_bm25_by_label(df: pd.DataFrame) -> dict:
    print_section("Test 5 — BM25 Score Distribution by Relevance Label (Kruskal-Wallis)")
    label_order = ["E", "S", "C", "I"]
    label_names = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}
    groups = {l: df[df["esci_label"] == l]["bm25_score"].values for l in label_order}

    print(f"\n  H0: BM25 score distribution identical across all relevance labels")
    print(f"  H1: At least one label has a different BM25 score distribution\n")
    for l in label_order:
        g = groups[l]
        print(f"  {label_names[l]:<14}  n={len(g):>6,}   median={np.median(g):.4f}   mean={g.mean():.4f}")

    stat, p   = stats.kruskal(*[groups[l] for l in label_order])
    n         = len(df)
    eta_sq    = (stat - len(label_order) + 1) / (n - len(label_order))
    sig       = p < ALPHA

    print(f"\n  Kruskal-Wallis H:  {stat:.4f}")
    print(f"  p-value:           {p:.2e}")
    print(f"  Eta-squared:       {eta_sq:.4f}  ({'large' if eta_sq > 0.14 else 'medium' if eta_sq > 0.06 else 'small'} effect)")
    print(f"\n  {'REJECT H0' if sig else 'FAIL TO REJECT H0'} — BM25 scores differ {'significantly' if sig else 'NOT significantly'} across labels.")

    bonferroni_alpha = ALPHA / 6
    pairs = [(l1, l2) for i, l1 in enumerate(label_order) for l2 in label_order[i+1:]]
    print(f"\n  Post-hoc Mann-Whitney U (Bonferroni alpha = {bonferroni_alpha:.4f}):")
    pairwise = {}
    for l1, l2 in pairs:
        u_stat, p_raw = stats.mannwhitneyu(groups[l1], groups[l2], alternative="two-sided")
        p_corr  = min(p_raw * 6, 1.0)
        sig_pair = p_corr < ALPHA
        pairwise[f"{label_names[l1]}_vs_{label_names[l2]}"] = {
            "u_stat": float(u_stat), "p_corrected": float(p_corr), "significant": bool(sig_pair)
        }
        marker = "*" if sig_pair else " "
        print(f"  {marker} {label_names[l1]:<14} vs {label_names[l2]:<14}  p={p_corr:.2e}  {'significant' if sig_pair else 'ns'}")

    return {
        "kruskal_h": float(stat), "p_value": float(p), "eta_squared": float(eta_sq),
        "significant": bool(sig), "pairwise": pairwise,
    }


def run_test6_query_length(df: pd.DataFrame) -> dict:
    print_section("Test 6 — Query Length vs BM25 NDCG@10 (Point-Biserial Correlation)")

    query_meta = df.groupby("query_id").first().reset_index()
    query_meta["query_len"] = query_meta["query"].str.split().str.len()

    per_query = df.groupby("query_id").apply(
        lambda g: ndcg_at_k(
            g.sort_values("bm25_score", ascending=False)["gain"].tolist(),
            g["gain"].tolist(), 10
        )
    ).reset_index()
    per_query.columns = ["query_id", "ndcg_10"]
    merged = query_meta.merge(per_query, on="query_id")

    rho, p_spear = stats.spearmanr(merged["query_len"], merged["ndcg_10"])

    short  = merged[merged["query_len"] <= 2]["ndcg_10"].values
    medium = merged[(merged["query_len"] >= 3) & (merged["query_len"] <= 4)]["ndcg_10"].values
    long_  = merged[merged["query_len"] >= 5]["ndcg_10"].values

    print(f"\n  H0: Query length has no correlation with BM25 NDCG@10")
    print(f"  H1: Query length is correlated with BM25 NDCG@10\n")
    print(f"  Short  (<=2 tokens)  n={len(short):>4}   mean NDCG@10 = {short.mean():.4f}")
    print(f"  Medium (3-4 tokens)  n={len(medium):>4}   mean NDCG@10 = {medium.mean():.4f}")
    print(f"  Long   (>=5 tokens)  n={len(long_):>4}   mean NDCG@10 = {long_.mean():.4f}")
    print(f"\n  Spearman rho: {rho:.4f}   p = {p_spear:.4e}")
    print(f"  {'REJECT H0' if p_spear < ALPHA else 'FAIL TO REJECT H0'} — query length {'does' if p_spear < ALPHA else 'does not'} significantly correlate with BM25 performance.")

    kw_stat, kw_p = stats.kruskal(short, medium, long_)
    print(f"\n  Kruskal-Wallis across length groups: H = {kw_stat:.4f},  p = {kw_p:.4e}")

    return {
        "spearman_rho": float(rho), "p_value": float(p_spear), "significant": bool(p_spear < ALPHA),
        "group_means": {
            "short_lte2": float(short.mean()), "medium_3to4": float(medium.mean()), "long_gte5": float(long_.mean())
        },
        "kruskal": {"h_stat": float(kw_stat), "p_value": float(kw_p)},
    }


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print_section("Loading Data")
    df = load_test_data()

    print_section("Computing BM25 Scores")
    df = add_bm25_scores(df)

    print_section("Computing ColBERT Scores")
    _download(COLBERT_CKPT, "/tmp/colbert_hyp.pt")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})
    colbert = ColBERT(dim=COLBERT_DIM)
    colbert.bert.resize_token_embeddings(len(tokenizer))
    colbert.load_state_dict(torch.load("/tmp/colbert_hyp.pt", map_location=device))
    colbert.to(device).eval()
    print(f"  ColBERT loaded ({device}). Running inference...")
    df = add_colbert_scores(df, colbert, tokenizer, device)

    bm25_ndcg = compute_per_query_ndcg(df, "bm25_score")
    cb_ndcg   = compute_per_query_ndcg(df, "colbert_score")

    results = {}
    results["test1_normality"]         = run_test1_normality(bm25_ndcg, cb_ndcg)
    results["test2_colbert_vs_bm25"]   = run_test2_colbert_vs_bm25(bm25_ndcg, cb_ndcg)
    results["test3_ndcg_cutoffs"]      = run_test3_ndcg_cutoffs(df)
    results["test4_spearman"]          = run_test4_spearman(df)
    results["test5_bm25_by_label"]     = run_test5_bm25_by_label(df)
    results["test6_query_length"]      = run_test6_query_length(df)

    print_section("Summary")
    rows = [
        ("Shapiro-Wilk: NDCG@10 normality",          results["test1_normality"]["BM25"]["normal"] and results["test1_normality"]["ColBERT"]["normal"]),
        ("ColBERT > BM25 (paired t-test)",            results["test2_colbert_vs_bm25"]["ttest"]["significant"]),
        ("ColBERT > BM25 (Wilcoxon)",                 results["test2_colbert_vs_bm25"]["wilcoxon"]["significant"]),
        ("ColBERT > BM25 @ NDCG@1",                   results["test3_ndcg_cutoffs"]["ndcg_1"]["significant"]),
        ("ColBERT > BM25 @ NDCG@5",                   results["test3_ndcg_cutoffs"]["ndcg_5"]["significant"]),
        ("BM25/ColBERT rankings correlated (rho!=0)", results["test4_spearman"]["significant"]),
        ("BM25 scores differ by relevance label",     results["test5_bm25_by_label"]["significant"]),
        ("Query length correlates with BM25 NDCG",    results["test6_query_length"]["significant"]),
    ]
    print(f"\n  {'Finding':<48} {'Result'}")
    print(f"  {'-' * 64}")
    for label, sig in rows:
        print(f"  {label:<48} {'YES *' if sig else 'no':>6}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "hypothesis_tests.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
