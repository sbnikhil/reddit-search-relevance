import os
import yaml

# Load .env file if present — must happen before any os.getenv() calls
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

_cfg_path = os.path.join(os.path.dirname(__file__), "settings.yaml")
with open(_cfg_path) as _f:
    _yaml = yaml.safe_load(_f)

# ── GCP ──────────────────────────────────────────────────────────────────────
# Override any value with env vars — only .env needs to change when migrating accounts.
PROJECT_ID    = os.getenv("GCP_PROJECT_ID",     _yaml["gcp"]["project_id"])
REGION        = os.getenv("GCP_REGION",          _yaml["gcp"]["region"])
DATA_BUCKET   = os.getenv("GCP_DATA_BUCKET",     _yaml["gcp"]["data_bucket"])
MODELS_BUCKET = os.getenv("GCP_MODELS_BUCKET",   _yaml["gcp"]["models_bucket"])
BQ_DATASET    = os.getenv("BQ_DATASET",          _yaml["gcp"]["bq_dataset"])

# ── GCS paths (built from bucket + relative path in yaml) ────────────────────
COLBERT_CKPT_GCS    = f"gs://{MODELS_BUCKET}/colbert/{_yaml['colbert']['best_checkpoint']}"
LAMBDAMART_CKPT_GCS = f"gs://{MODELS_BUCKET}/{_yaml['lambdamart']['model_path']}"

# ── ColBERT ───────────────────────────────────────────────────────────────────
COLBERT_MODEL_NAME   = _yaml["colbert"]["model_name"]
COLBERT_DIM          = _yaml["colbert"]["dim"]
COLBERT_MAXLEN_QUERY = _yaml["colbert"]["query_maxlen"]
COLBERT_MAXLEN_DOC   = _yaml["colbert"]["doc_maxlen"]
COLBERT_TEMPERATURE  = _yaml["colbert"]["temperature"]
COLBERT_WARMUP_STEPS = _yaml["colbert"]["warmup_steps"]
COLBERT_BATCH_SIZE   = _yaml["colbert"]["batch_size"]
COLBERT_EPOCHS       = _yaml["colbert"]["epochs"]
COLBERT_LR           = _yaml["colbert"]["learning_rate"]

# ── LambdaMART ───────────────────────────────────────────────────────────────
LTR_NUM_BOOST_ROUND       = _yaml["lambdamart"]["num_boost_round"]
LTR_EARLY_STOPPING_ROUNDS = _yaml["lambdamart"]["early_stopping_rounds"]

# Full params dict passed directly to lgb.train() — no hardcoding in model files
LAMBDAMART_PARAMS = {
    "objective":        "lambdarank",
    "metric":           "ndcg",
    "ndcg_eval_at":     _yaml["lambdamart"]["ndcg_eval_at"],
    "label_gain":       _yaml["lambdamart"]["label_gain"],
    "learning_rate":    _yaml["lambdamart"]["learning_rate"],
    "num_leaves":       _yaml["lambdamart"]["num_leaves"],
    "min_data_in_leaf": _yaml["lambdamart"]["min_data_in_leaf"],
    "feature_fraction": _yaml["lambdamart"]["feature_fraction"],
    "bagging_fraction": _yaml["lambdamart"]["bagging_fraction"],
    "bagging_freq":     _yaml["lambdamart"]["bagging_freq"],
    "lambda_l1":        _yaml["lambdamart"]["lambda_l1"],
    "lambda_l2":        _yaml["lambdamart"]["lambda_l2"],
    "verbose":          -1,
}

# ── Retrieval ─────────────────────────────────────────────────────────────────
BM25_TOP_K     = _yaml["retrieval"]["bm25_top_k"]
COLBERT_TOP_K  = _yaml["retrieval"]["colbert_top_k"]
FINAL_TOP_K    = _yaml["retrieval"]["final_top_k"]

# ── Training ──────────────────────────────────────────────────────────────────
HARD_NEG_TOP_K              = _yaml["training"]["hard_neg_top_k"]
HARD_NEG_RANK_THRESHOLD     = _yaml["training"]["hard_neg_rank_threshold"]
SCORE_GEN_DOC_BATCH         = _yaml["training"]["score_gen_doc_batch"]
SCORE_GEN_CHECKPOINT_EVERY  = _yaml["training"]["score_gen_checkpoint_every"]

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_SAMPLE_QUERIES = _yaml["evaluation"]["sample_queries"]
EVAL_DOC_BATCH      = _yaml["evaluation"]["doc_batch"]
EVAL_ALPHA          = _yaml["evaluation"]["alpha"]

# ── Serving ───────────────────────────────────────────────────────────────────
SERVING_HOST              = _yaml["serving"]["host"]
SERVING_PORT              = _yaml["serving"]["port"]
SERVING_DOC_BATCH         = _yaml["serving"]["doc_encode_batch"]
SERVING_DEMO_CATALOG_SIZE = _yaml["serving"]["demo_catalog_size"]
SERVING_USE_QUERY_ROUTER = _yaml["serving"]["use_query_router"]

# ── Contribution 1: Query-conditioned LTR (reference models, not in prod path) ─
CONTRIB1_RHO_LOW        = _yaml["contribution1"]["rho_low_threshold"]
CONTRIB1_RHO_HIGH       = _yaml["contribution1"]["rho_high_threshold"]
CONTRIB1_MODEL_LOW_GCS  = f"gs://{MODELS_BUCKET}/{_yaml['contribution1']['model_low']}"
CONTRIB1_MODEL_MED_GCS  = f"gs://{MODELS_BUCKET}/{_yaml['contribution1']['model_medium']}"
CONTRIB1_MODEL_HIGH_GCS = f"gs://{MODELS_BUCKET}/{_yaml['contribution1']['model_high']}"

# ── FAISS dense retrieval index (built by script 15) ─────────────────────────
FAISS_INDEX_GCS   = f"gs://{MODELS_BUCKET}/faiss/product_index.faiss"
FAISS_IDS_GCS     = f"gs://{MODELS_BUCKET}/faiss/product_id_map.json"
FAISS_N_CENTROIDS = 2048
FAISS_NPROBE      = 32
