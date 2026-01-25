import torch
import pysolr
import numpy as np
import yaml
import time
import re
import os
from google.cloud import bigquery
from transformers import AutoTokenizer
from models.arch.relevance_ranker import RedditRelevanceRanker

# 1. SETUP & CONFIG
with open("config/settings.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Load feature normalization parameters
import json

norm_params_path = os.path.join(cfg["artifacts"]["model_path"], "feature_norm_params.json")
if os.path.exists(norm_params_path):
    with open(norm_params_path, "r") as f:
        norm_params = json.load(f)
    print(f"Loaded feature normalization params: {norm_params}")
else:
    print("Warning: No normalization params found, using defaults [0,1]")
    norm_params = {"expertise_min": 0, "expertise_max": 1, "utility_min": 0, "utility_max": 1}

try:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
except:
    device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(cfg["training"]["model_name"])
bq_client = bigquery.Client(project=cfg["database"]["project_id"])
solr = pysolr.Solr(cfg["search_tuning"]["solr_url"], timeout=10)

# 2. LOAD MODEL
model = RedditRelevanceRanker(
    cfg["training"]["model_name"],
    cfg["training"]["extra_feature_dim"],
    dropout=cfg["model_params"].get("dropout_rate", 0.1),
).to(device)

model_file = f"reddit_ranker_v{cfg['artifacts']['version']}.pt"
model_path = os.path.join(cfg["artifacts"]["model_path"], model_file)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"loaded model: {model_file}")
else:
    print(f"error loading model!")
model.eval()


def calculate_ndcg(binary_relevance):
    """Calculates nDCG @ K using the standard formula."""
    if not binary_relevance or sum(binary_relevance) == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i + 2))
    dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(binary_relevance)])

    # IDCG = sum(1 / log2(i + 2)) for the number of relevant items found
    num_rel = int(sum(binary_relevance))
    idcg = sum([1.0 / np.log2(i + 2) for i in range(num_rel)])

    return dcg / idcg


def evaluate():
    # Use a large enough offset to avoid training data leakage
    sql = f"SELECT body FROM `{cfg['database']['source_table']}` WHERE label = 1 LIMIT 200 OFFSET 5000"
    df = bq_client.query(sql).to_dataframe()

    stats = {"s_ndcg": [], "h_ndcg": [], "recall": [], "latencies": []}

    print(f"Analyzing {len(df)} search queries...")

    for idx, body in enumerate(df["body"].tolist()):
        # Generate query from first 8 words
        raw_q = " ".join(re.sub("<.*?>", "", body).split()[:8])
        q_clean = re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r"\\\1", raw_q)

        start = time.perf_counter()
        try:
            # A. SOLR RETRIEVAL
            res = solr.search(f"body_t:({q_clean})", rows=cfg["search_tuning"]["k_recall"], fl="*,score")
            if not res or len(res) < 2:
                continue

            # B. GROUND TRUTH (Use Solr score as proxy - top 10 results are considered relevant)
            # This assumes Solr's BM25 scoring provides reasonable relevance signal
            solr_scores_raw = [float(d.get("score", 0.0)) for d in res]
            score_threshold = (
                sorted(solr_scores_raw, reverse=True)[min(9, len(solr_scores_raw) - 1)]
                if len(solr_scores_raw) > 10
                else 0
            )
            binary_rel = [1 if score >= score_threshold else 0 for score in solr_scores_raw]

            # C. BERT RE-RANKING
            batch_docs = [d.get("body_t", "") for d in res]
            inputs = tokenizer(
                [raw_q] * len(res), batch_docs, return_tensors="pt", truncation=True, padding=True, max_length=128
            ).to(device)

            # Pull expertise/utility features from Solr (if available) or use defaults
            raw_features = [[float(d.get("expertise_score", 0.0)), float(d.get("utility_score", 0.0))] for d in res]

            # Normalize features using training-time scale
            normalized_features = []
            for exp, util in raw_features:
                exp_norm = (exp - norm_params["expertise_min"]) / (
                    norm_params["expertise_max"] - norm_params["expertise_min"] + 1e-6
                )
                util_norm = (util - norm_params["utility_min"]) / (
                    norm_params["utility_max"] - norm_params["utility_min"] + 1e-6
                )
                # Clip to [0, 1] in case of out-of-distribution values
                exp_norm = max(0.0, min(1.0, exp_norm))
                util_norm = max(0.0, min(1.0, util_norm))
                normalized_features.append([exp_norm, util_norm])

            feats = torch.tensor(normalized_features, dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model(inputs["input_ids"], inputs["attention_mask"], feats)
                b_scores = torch.sigmoid(logits).cpu().numpy().flatten()

            # D. LOGIC: SCORE NORMALIZATION & BLENDING
            s_scores = np.array(solr_scores_raw)

            # Min-Max Scale Solr to 0-1
            s_min, s_max = s_scores.min(), s_scores.max()
            s_norm = (s_scores - s_min) / (s_max - s_min + 1e-6)

            # Hybrid Formula from Config
            alpha = cfg["search_tuning"]["alpha"]
            # Both scores now in [0, 1] range - linear combination
            h_scores = (alpha * s_norm) + ((1 - alpha) * b_scores)

            # E. METRICS
            # Solr Baseline
            stats["s_ndcg"].append(calculate_ndcg(binary_rel))

            # Hybrid Ranking (Sort results by h_scores descending)
            h_ranked_rel = [rel for _, rel in sorted(zip(h_scores, binary_rel), key=lambda x: x[0], reverse=True)]
            stats["h_ndcg"].append(calculate_ndcg(h_ranked_rel))

            stats["recall"].append(1 if sum(binary_rel) > 0 else 0)
            stats["latencies"].append((time.perf_counter() - start) * 1000)

        except Exception as e:
            continue

    # 3. FINAL SUMMARY
    s_avg = np.mean(stats["s_ndcg"])
    h_avg = np.mean(stats["h_ndcg"])
    lift = ((h_avg - s_avg) / (s_avg + 1e-9)) * 100

    print(f"\n" + "=" * 40)
    print(f"FINAL EVALUATION RESULTS")
    print("=" * 40)
    print(f"Solr nDCG:   {s_avg:.4f}")
    print(f"Hybrid nDCG: {h_avg:.4f}")
    print(f"LIFT:        {lift:+.2f}%")
    print(f"Recall@K:    {np.mean(stats['recall'])*100:.1f}%")
    print(f"P99 Latency: {np.percentile(stats['latencies'], 99):.1f}ms")
    print("=" * 40)


if __name__ == "__main__":
    evaluate()
