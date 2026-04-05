import os
import time
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from transformers import BertTokenizerFast

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.colbert.architecture import ColBERT
from models.ltr.lambdamart import LambdaMARTRanker, build_features

_cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "settings.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

PROJECT_ID      = os.environ.get("GCP_PROJECT_ID",    _cfg["gcp"]["project_id"])
MODELS_BUCKET   = os.environ.get("GCP_MODEL_BUCKET",  _cfg["gcp"]["models_bucket"])
DATA_BUCKET     = os.environ.get("GCP_DATA_BUCKET",   _cfg["gcp"]["data_bucket"])
COLBERT_CKPT    = os.environ.get("COLBERT_CHECKPOINT", f"gs://{MODELS_BUCKET}/colbert/epoch_5/model.pt")
LTR_MODEL       = os.environ.get("LAMBDAMART_MODEL",   f"gs://{MODELS_BUCKET}/ltr/lambdamart.txt")
CATALOG_GCS     = f"gs://{DATA_BUCKET}/esci/catalog/products.parquet"
COLBERT_DIM     = _cfg["colbert"]["dim"]
QUERY_MAXLEN    = _cfg["colbert"]["query_maxlen"]
DOC_MAXLEN      = _cfg["colbert"]["doc_maxlen"]
BM25_TOP_K      = _cfg["retrieval"]["bm25_top_k"]
DOC_ENCODE_BATCH = 64

_state: dict = {}
_latencies: list = []
_metrics: dict = {"ndcg_10": 0.0, "mrr_10": 0.0, "p99_latency_ms": 0.0}


def _download_gcs(gcs_uri: str, local_path: str) -> None:
    client = storage.Client()
    bucket, blob = gcs_uri.replace("gs://", "").split("/", 1)
    client.bucket(bucket).blob(blob).download_to_filename(local_path)


def _load_catalog() -> tuple:
    """Load product catalog. Returns (product_ids, titles, descriptions, brands, bm25_index)."""
    import pandas as pd

    local_parquet = "/tmp/products_catalog.parquet"
    if not os.path.exists(local_parquet):
        try:
            print(f"Downloading catalog from {CATALOG_GCS}...")
            _download_gcs(CATALOG_GCS, local_parquet)
        except Exception:
            print("GCS catalog not found -- querying BigQuery (slow first start)...")
            from google.cloud import bigquery
            bq = bigquery.Client(project=PROJECT_ID)
            df = bq.query(f"""
                SELECT product_id,
                       COALESCE(product_title, '')       AS product_title,
                       COALESCE(product_description, '') AS product_description,
                       COALESCE(product_brand, '')       AS product_brand
                FROM `{PROJECT_ID}.esci_search.products`
            """).to_dataframe()
            df.to_parquet(local_parquet, index=False)
            # Cache to GCS for next start
            try:
                bucket, blob = CATALOG_GCS.replace("gs://", "").split("/", 1)
                storage.Client().bucket(bucket).blob(blob).upload_from_filename(local_parquet)
                print(f"Catalog cached -> {CATALOG_GCS}")
            except Exception as e:
                print(f"Warning: could not cache catalog to GCS: {e}")

    df = pd.read_parquet(local_parquet)
    print(f"Loaded {len(df):,} products.")

    product_ids   = df["product_id"].tolist()
    titles        = df["product_title"].tolist()
    descriptions  = df["product_description"].tolist()
    brands        = df["product_brand"].tolist()

    print("Building BM25 index...")
    tokenized = [t.lower().split() for t in titles]
    bm25 = BM25Okapi(tokenized)

    return product_ids, titles, descriptions, brands, bm25


def _load_colbert(device: torch.device) -> tuple[ColBERT, BertTokenizerFast]:
    local = "/tmp/colbert_serving.pt"
    _download_gcs(COLBERT_CKPT, local)
    model = ColBERT(dim=COLBERT_DIM)
    model.load_state_dict(torch.load(local, map_location=device))
    model.to(device).eval()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    print(f"ColBERT loaded ({device}).")
    return model, tokenizer


def _load_ltr() -> LambdaMARTRanker:
    ranker = LambdaMARTRanker()
    ranker.load(LTR_MODEL)
    print("LambdaMART loaded.")
    return ranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    product_ids, titles, descriptions, brands, bm25 = _load_catalog()
    colbert_model, tokenizer = _load_colbert(device)
    ltr = _load_ltr()

    catalog = {
        pid: {"title": t, "description": d, "brand": b}
        for pid, t, d, b in zip(product_ids, titles, descriptions, brands)
    }

    _state.update({
        "device":        device,
        "product_ids":   product_ids,
        "bm25":          bm25,
        "catalog":       catalog,
        "colbert":       colbert_model,
        "tokenizer":     tokenizer,
        "ltr":           ltr,
    })
    print("Serving ready.")
    yield
    _state.clear()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class StageScores(BaseModel):
    bm25: float
    colbert: float
    ltr: float


class SearchResult(BaseModel):
    product_id: str
    title: str
    score: float
    stage_scores: StageScores


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    latency_ms: float


app = FastAPI(title="ESCI Search API", lifespan=lifespan)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not _state:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    t0 = time.perf_counter()
    query = request.query.strip()
    top_k = max(1, min(request.top_k, BM25_TOP_K))

    device      = _state["device"]
    product_ids = _state["product_ids"]
    bm25        = _state["bm25"]
    catalog     = _state["catalog"]
    colbert     = _state["colbert"]
    tokenizer   = _state["tokenizer"]
    ltr         = _state["ltr"]

    q_tokens  = query.lower().split()
    bm25_raw  = bm25.get_scores(q_tokens)                          # (N,)
    k         = min(BM25_TOP_K, len(bm25_raw))
    top_idxs  = np.argpartition(bm25_raw, -k)[-k:]
    top_idxs  = top_idxs[np.argsort(bm25_raw[top_idxs])[::-1]]

    cand_ids    = [product_ids[i] for i in top_idxs]
    cand_bm25   = bm25_raw[top_idxs]
    bm25_max    = cand_bm25.max() or 1.0
    cand_bm25_n = (cand_bm25 / bm25_max).tolist()

    cand_titles = [catalog[pid]["title"]       for pid in cand_ids]
    cand_descs  = [catalog[pid]["description"] for pid in cand_ids]
    cand_brands = [catalog[pid]["brand"]       for pid in cand_ids]

    q_enc = tokenizer(
        [query], padding=True, truncation=True,
        max_length=QUERY_MAXLEN, return_tensors="pt",
    )
    with torch.no_grad():
        q_emb = colbert.encode_query(
            q_enc["input_ids"].to(device),
            q_enc["attention_mask"].to(device),
        )[0]  # (Lq, dim)

    colbert_scores = []
    for start in range(0, len(cand_titles), DOC_ENCODE_BATCH):
        batch = cand_titles[start : start + DOC_ENCODE_BATCH]
        d_enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=DOC_MAXLEN, return_tensors="pt",
        )
        with torch.no_grad():
            d_embs = colbert.encode_document(
                d_enc["input_ids"].to(device),
                d_enc["attention_mask"].to(device),
            )  # (B, Ld, dim)
        sim    = torch.einsum("qd,bld->bql", q_emb, d_embs)
        scores = sim.max(dim=2).values.sum(dim=1).cpu().numpy()
        colbert_scores.extend(scores.tolist())

    cb_arr    = np.array(colbert_scores)
    cb_max    = cb_arr.max() or 1.0
    cb_norm   = (cb_arr / cb_max).tolist()

    features = np.array([
        build_features(query, t, d, b, bm25_s, cb_s)
        for t, d, b, bm25_s, cb_s
        in zip(cand_titles, cand_descs, cand_brands, cand_bm25_n, cb_norm)
    ], dtype=np.float32)

    ltr_scores = ltr.predict(features)

    order = np.argsort(ltr_scores)[::-1][:top_k]
    results = [
        SearchResult(
            product_id=cand_ids[i],
            title=cand_titles[i],
            score=float(ltr_scores[i]),
            stage_scores=StageScores(
                bm25=cand_bm25_n[i],
                colbert=cb_norm[i],
                ltr=float(ltr_scores[i]),
            ),
        )
        for i in order
    ]

    latency_ms = (time.perf_counter() - t0) * 1000
    _latencies.append(latency_ms)
    if len(_latencies) > 1000:
        _latencies.pop(0)
    if _latencies:
        _metrics["p99_latency_ms"] = float(np.percentile(_latencies, 99))

    return SearchResponse(results=results, query=query, latency_ms=latency_ms)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "colbert+lambdamart", "ready": bool(_state)}


@app.get("/metrics")
async def metrics():
    return _metrics


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=_cfg["serving"]["host"], port=_cfg["serving"]["port"])
