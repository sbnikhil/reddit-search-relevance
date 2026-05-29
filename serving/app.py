"""
ESCI Search API — new architecture.

Stage 1: Hybrid Retrieval
  BM25 top-50 + FAISS IVFFlat top-50 → Reciprocal Rank Fusion (k=60) → top-100
  Query router (token count rule) skips FAISS for long queries (≥5 tokens).

Stage 2: ColBERT MaxSim re-ranking
  Re-ranks the top-100 RRF candidates.
  Cross-encoder (Contribution 1) will replace this once trained.

Feature flags (env vars):
  USE_QUERY_ROUTER=true|false
  COLBERT_CHECKPOINT=gs://...
"""
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from transformers import BertTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROJECT_ID, BQ_DATASET, MODELS_BUCKET, DATA_BUCKET,
    COLBERT_DIM, COLBERT_MODEL_NAME, COLBERT_MAXLEN_QUERY, COLBERT_MAXLEN_DOC,
    BM25_TOP_K, COLBERT_CKPT_GCS,
    FAISS_INDEX_GCS, FAISS_IDS_GCS, FAISS_NPROBE,
    SERVING_HOST, SERVING_PORT, SERVING_DOC_BATCH,
    SERVING_USE_QUERY_ROUTER, SERVING_DEMO_CATALOG_SIZE,
)
from models.colbert.architecture import ColBERT
from models.query_router import QueryRouter
from utils.gcs import download

COLBERT_CKPT      = os.environ.get("COLBERT_CHECKPOINT", COLBERT_CKPT_GCS)
DOC_ENCODE_BATCH  = SERVING_DOC_BATCH
USE_QUERY_ROUTER  = os.environ.get("USE_QUERY_ROUTER", str(SERVING_USE_QUERY_ROUTER)).lower() == "true"
DEMO_CATALOG_SIZE = int(os.environ.get("DEMO_CATALOG_SIZE", str(SERVING_DEMO_CATALOG_SIZE)))
CATALOG_GCS       = f"gs://{DATA_BUCKET}/esci/catalog/products.parquet"
RRF_K             = 60
FAISS_TOP_K       = 50
BM25_RETRIEVAL_K  = 50

_state: dict     = {}
_latencies: list = []


# ── Startup loaders ───────────────────────────────────────────────────────────

def _load_catalog() -> tuple:
    import pandas as pd
    local_parquet = "/tmp/products_catalog.parquet"
    if not os.path.exists(local_parquet):
        try:
            download(CATALOG_GCS, local_parquet)
        except Exception:
            print("GCS catalog not found — querying BigQuery...")
            from google.cloud import bigquery
            bq = bigquery.Client(project=PROJECT_ID)
            df = bq.query(f"""
                SELECT product_id,
                       COALESCE(product_title, '')       AS product_title,
                       COALESCE(product_description, '') AS product_description,
                       COALESCE(product_brand, '')       AS product_brand
                FROM `{PROJECT_ID}.{BQ_DATASET}.products`
            """).to_dataframe()
            df.to_parquet(local_parquet, index=False)

    df = pd.read_parquet(local_parquet)
    if DEMO_CATALOG_SIZE > 0:
        df = df.head(DEMO_CATALOG_SIZE)
        print(f"DEMO MODE: catalog capped at {DEMO_CATALOG_SIZE:,} products.")
    print(f"Loaded {len(df):,} products.")
    catalog = {
        row["product_id"]: {
            "title":       row["product_title"],
            "description": row["product_description"],
            "brand":       row["product_brand"],
        }
        for _, row in df.iterrows()
    }
    print("Building BM25 index...")
    product_ids = df["product_id"].tolist()
    bm25 = BM25Okapi(
        [(row["product_title"] + " " + row["product_description"]).lower().split()
         for _, row in df.iterrows()]
    )
    return product_ids, bm25, catalog


def _load_faiss() -> Optional[tuple]:
    try:
        local_index = "/tmp/serving_product_index.faiss"
        local_idmap = "/tmp/serving_product_id_map.json"
        download(FAISS_INDEX_GCS, local_index)
        download(FAISS_IDS_GCS,   local_idmap)
        index = faiss.read_index(local_index)
        index.nprobe = FAISS_NPROBE
        with open(local_idmap) as f:
            faiss_ids = json.load(f)
        print(f"FAISS index loaded: {index.ntotal:,} vectors")
        return index, faiss_ids
    except Exception as e:
        print(f"WARNING: FAISS index not available ({e}). Falling back to BM25-only.")
        return None


def _load_colbert(device: torch.device) -> tuple:
    local = "/tmp/colbert_serving.pt"
    download(COLBERT_CKPT, local)
    tokenizer = BertTokenizerFast.from_pretrained(COLBERT_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})
    model = ColBERT(dim=COLBERT_DIM)
    model.bert.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(local, map_location=device, weights_only=False))
    model.to(device).eval()
    print(f"ColBERT loaded ({device}). Vocab: {len(tokenizer)}")
    return model, tokenizer


# ── Request / Response ────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class StageScores(BaseModel):
    bm25:    float
    colbert: float


class SearchResult(BaseModel):
    product_id:   str
    title:        str
    score:        float
    stage_scores: StageScores


class SearchResponse(BaseModel):
    results:      List[SearchResult]
    query:        str
    latency_ms:   float
    retrieval:    str   # "bm25_faiss_rrf" | "bm25_only"
    n_candidates: int


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting up on device: {device}")

    product_ids, bm25, catalog  = _load_catalog()
    faiss_components            = _load_faiss()
    colbert_model, tokenizer    = _load_colbert(device)
    query_router                = QueryRouter() if USE_QUERY_ROUTER else None

    # Build a fast lookup: catalog product_id → bm25 position
    pid_to_bm25_idx = {pid: i for i, pid in enumerate(product_ids)}

    _state.update({
        "device":          device,
        "product_ids":     product_ids,
        "pid_to_bm25_idx": pid_to_bm25_idx,
        "bm25":            bm25,
        "catalog":         catalog,
        "faiss":           faiss_components,
        "colbert":         colbert_model,
        "tokenizer":       tokenizer,
        "query_router":    query_router,
    })
    print("Serving ready.")
    yield
    _state.clear()


app = FastAPI(title="ESCI Search API", lifespan=lifespan)


# ── Search endpoint ───────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not _state:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    t0    = time.perf_counter()
    query = request.query.strip()
    top_k = max(1, min(request.top_k, 100))

    device         = _state["device"]
    product_ids    = _state["product_ids"]
    pid_to_bm25    = _state["pid_to_bm25_idx"]
    bm25           = _state["bm25"]
    catalog        = _state["catalog"]
    faiss_comp     = _state["faiss"]
    colbert        = _state["colbert"]
    tokenizer      = _state["tokenizer"]
    query_router   = _state["query_router"]

    # ── Stage 1: Hybrid Retrieval ─────────────────────────────────────────────
    use_faiss = (
        faiss_comp is not None and
        (query_router is None or query_router.use_faiss(query))
    )

    # BM25 retrieval
    raw_bm25   = bm25.get_scores(query.lower().split())
    k_bm25     = min(BM25_RETRIEVAL_K, len(raw_bm25))
    bm25_top   = np.argpartition(raw_bm25, -k_bm25)[-k_bm25:]
    bm25_top   = bm25_top[np.argsort(raw_bm25[bm25_top])[::-1]]
    bm25_pids  = [product_ids[i] for i in bm25_top]
    bm25_sdict = {product_ids[i]: raw_bm25[i] for i in bm25_top}

    if use_faiss:
        faiss_index, faiss_ids = faiss_comp
        q_enc = tokenizer(
            [f"[Q] {query}"], max_length=COLBERT_MAXLEN_QUERY,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        with torch.no_grad():
            q_emb = colbert.encode_query(
                q_enc["input_ids"].to(device),
                q_enc["attention_mask"].to(device),
            )[:, 0, :].cpu().numpy().astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = faiss_index.search(q_emb, FAISS_TOP_K)
        faiss_pids = [faiss_ids[j] for j in I[0] if j >= 0]

        # RRF merge
        rrf_scores: dict[str, float] = {}
        for rank, pid in enumerate(bm25_pids):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, pid in enumerate(faiss_pids):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        cand_pids   = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)[:100]
        retrieval_mode = "bm25_faiss_rrf"
    else:
        cand_pids   = bm25_pids[:100]
        retrieval_mode = "bm25_only"

    # Normalize BM25 scores for ColBERT feature
    bm25_arr = np.array([bm25_sdict.get(pid, 0.0) for pid in cand_pids])
    bm25_max = bm25_arr.max() if bm25_arr.max() > 0 else 1.0
    bm25_norm = (bm25_arr / bm25_max).tolist()

    cand_titles = [catalog[pid]["title"]       for pid in cand_pids if pid in catalog]
    cand_descs  = [catalog[pid]["description"] for pid in cand_pids if pid in catalog]
    cand_pids   = [pid for pid in cand_pids if pid in catalog]

    # ── Stage 2: ColBERT MaxSim re-ranking ───────────────────────────────────
    q_enc = tokenizer(
        [f"[Q] {query}"], padding=True, truncation=True,
        max_length=COLBERT_MAXLEN_QUERY, return_tensors="pt",
    )
    with torch.no_grad():
        q_emb_full = colbert.encode_query(
            q_enc["input_ids"].to(device),
            q_enc["attention_mask"].to(device),
        )[0]

    cb_scores = []
    for start in range(0, len(cand_titles), DOC_ENCODE_BATCH):
        batch = [f"[D] {t}" for t in cand_titles[start : start + DOC_ENCODE_BATCH]]
        d_enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=COLBERT_MAXLEN_DOC, return_tensors="pt",
        )
        with torch.no_grad():
            d_embs = colbert.encode_document(
                d_enc["input_ids"].to(device),
                d_enc["attention_mask"].to(device),
            )
        sim = torch.einsum("qd,bld->bql", q_emb_full, d_embs)
        cb_scores.extend(sim.max(dim=2).values.sum(dim=1).cpu().numpy().tolist())

    cb_arr  = np.array(cb_scores)
    cb_norm = (cb_arr / (cb_arr.max() or 1.0)).tolist()

    order = np.argsort(cb_arr)[::-1][:top_k]
    results = [
        SearchResult(
            product_id=cand_pids[i],
            title=cand_titles[i],
            score=float(cb_arr[i]),
            stage_scores=StageScores(bm25=bm25_norm[i], colbert=cb_norm[i]),
        )
        for i in order
    ]

    latency_ms = (time.perf_counter() - t0) * 1000
    _latencies.append(latency_ms)
    if len(_latencies) > 1000:
        _latencies.pop(0)

    return SearchResponse(
        results=results,
        query=query,
        latency_ms=latency_ms,
        retrieval=retrieval_mode,
        n_candidates=len(cand_pids),
    )


@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "ready":       bool(_state),
        "retrieval":   "bm25_faiss_rrf" if _state.get("faiss") else "bm25_only",
        "use_faiss":   _state.get("faiss") is not None,
        "query_router": _state.get("query_router") is not None,
    }


@app.get("/metrics")
async def metrics():
    p99 = float(np.percentile(_latencies, 99)) if _latencies else 0.0
    return {"p99_latency_ms": p99, "n_requests": len(_latencies)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", SERVING_PORT))
    uvicorn.run(app, host=SERVING_HOST, port=port)
