"""
Build FAISS IVFFlat index from ColBERT document [CLS] embeddings.

Encodes all ESCI-labeled products using the trained ColBERT document encoder,
extracts the [CLS] token embedding (dim=128, L2-normalised), and builds an
IVFFlat index for approximate nearest-neighbour retrieval at serving time.

Self-contained: ColBERT class, config, and GCS helpers are all inlined so this
script can run on Vertex AI without the full project package.

Output (GCS):
  search-models-nikhil/faiss/product_index.faiss
  search-models-nikhil/faiss/product_id_map.json

Usage:
  python scripts/15_build_faiss_index.py --submit     # Vertex AI full run
  python scripts/15_build_faiss_index.py --smoke      # 200 products, validate pipeline
  python scripts/15_build_faiss_index.py              # full local run (Linux only)
"""
import argparse
import json
import os
import sys
import time

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.cloud import bigquery, storage
from transformers import BertModel, BertTokenizerFast

# ── Config — env vars on Vertex AI, falls back to defaults locally ─────────────
PROJECT_ID         = os.environ.get("GCP_PROJECT_ID",      "search-relevance-500100")
BQ_DATASET         = os.environ.get("BQ_DATASET",          "esci_search")
MODELS_BUCKET      = os.environ.get("GCP_MODELS_BUCKET",   "search-models-nikhil")
REGION             = os.environ.get("GCP_REGION",          "us-central1")
COLBERT_MODEL_NAME = os.environ.get("COLBERT_MODEL_NAME",  "bert-base-uncased")
COLBERT_DIM        = int(os.environ.get("COLBERT_DIM",     "128"))
COLBERT_MAXLEN_DOC = int(os.environ.get("COLBERT_MAXLEN_DOC", "128"))
COLBERT_CKPT_GCS   = os.environ.get(
    "COLBERT_CHECKPOINT",
    f"gs://{MODELS_BUCKET}/colbert/epoch_5/model.pt",
)
FAISS_INDEX_GCS = f"gs://{MODELS_BUCKET}/faiss/product_index.faiss"
FAISS_IDS_GCS   = f"gs://{MODELS_BUCKET}/faiss/product_id_map.json"
N_CENTROIDS     = 2048
NPROBE          = 32


# ── Inlined ColBERT model ──────────────────────────────────────────────────────

class ColBERT(nn.Module):
    def __init__(self, model_name: str = COLBERT_MODEL_NAME, dim: int = COLBERT_DIM):
        super().__init__()
        self.bert   = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, dim)

    def encode_document(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embs = F.normalize(self.linear(out), p=2, dim=-1)
        return embs * attention_mask.unsqueeze(-1).float()


# ── Inlined GCS helpers ────────────────────────────────────────────────────────

def _gcs_download(gcs_uri: str, local_path: str) -> None:
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    print(f"Downloaded {gcs_uri} -> {local_path}")


def _gcs_upload(local_path: str, gcs_uri: str) -> None:
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> {gcs_uri}")


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> tuple:
    print(f"Loading ColBERT from {COLBERT_CKPT_GCS} ...")
    local_ckpt = "/tmp/colbert_faiss.pt"
    _gcs_download(COLBERT_CKPT_GCS, local_ckpt)

    tokenizer = BertTokenizerFast.from_pretrained(COLBERT_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})

    model = ColBERT(model_name=COLBERT_MODEL_NAME, dim=COLBERT_DIM)
    model.bert.resize_token_embeddings(len(tokenizer))

    state = torch.load(local_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  Model ready on {device}  vocab={len(tokenizer)}")
    return model, tokenizer


# ── Data loading ───────────────────────────────────────────────────────────────

def load_products(smoke: bool) -> list:
    client = bigquery.Client(project=PROJECT_ID)
    limit  = "LIMIT 200" if smoke else ""
    sql = f"""
    SELECT DISTINCT
        p.product_id,
        COALESCE(p.product_title, '')       AS product_title,
        COALESCE(p.product_description, '') AS product_description
    FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
    JOIN `{PROJECT_ID}.{BQ_DATASET}.products` p USING (product_id)
    ORDER BY product_id
    {limit}
    """
    print("Loading products from BigQuery...")
    df = client.query(sql).to_dataframe()
    print(f"  {len(df):,} products")
    return df.to_dict("records")


# ── Encoding ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_products(products, model, tokenizer, device, batch_size=64):
    product_ids    = []
    all_embeddings = []
    n  = len(products)
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch = products[start : start + batch_size]
        texts = [
            f"[D] {row['product_title']} {row['product_description']}"
            for row in batch
        ]
        enc = tokenizer(
            texts,
            max_length=COLBERT_MAXLEN_DOC,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        doc_embs = model.encode_document(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
        # CLS token (position 0) is never masked — unit-normalised by encode_document
        cls_embs = doc_embs[:, 0, :].cpu().numpy().astype("float32")
        all_embeddings.append(cls_embs)
        product_ids.extend(row["product_id"] for row in batch)

        if (start // batch_size) % 20 == 0:
            done    = start + len(batch)
            elapsed = time.time() - t0
            eta     = elapsed / done * (n - done) if elapsed > 1 else 0
            print(f"  {done:>7,}/{n:,}  {elapsed/60:.1f}m elapsed  ETA {eta/60:.1f}m")

    embeddings = np.vstack(all_embeddings)
    print(f"  Encoding done: {embeddings.shape}")
    return embeddings, product_ids


# ── Index building ─────────────────────────────────────────────────────────────

def build_index(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)

    # Use flat index for smoke (<500 vectors), IVFFlat for full run
    n = len(embeddings)
    if n < 500:
        print(f"Building IndexFlatIP (smoke mode, {n} vectors)...")
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
    else:
        n_centroids = min(N_CENTROIDS, n // 39)
        print(f"Building IVFFlat  d={d}  n_centroids={n_centroids}  n={n:,}")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_centroids, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = NPROBE

    print(f"  Index built: {index.ntotal:,} vectors")
    return index


def verify_index(index: faiss.Index, embeddings: np.ndarray, product_ids: list) -> None:
    if hasattr(index, "nprobe"):
        index.nprobe = NPROBE
    n_check = min(5, len(embeddings))
    D, I = index.search(embeddings[:n_check], 3)
    print("\n=== Sanity check (first neighbour should be self, score ~1.0) ===")
    for i in range(n_check):
        pid   = product_ids[i][:16]
        top1  = product_ids[I[i][0]][:16]
        score = D[i][0]
        match = "✓" if I[i][0] == i else "✗"
        print(f"  {pid}... → {top1}... score={score:.4f} {match}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model, tokenizer = load_model(device)
    products         = load_products(smoke=args.smoke)
    embeddings, pids = encode_products(products, model, tokenizer, device,
                                       batch_size=args.batch_size)

    index = build_index(embeddings)
    verify_index(index, embeddings, pids)

    local_index  = "/tmp/product_index.faiss"
    local_idmap  = "/tmp/product_id_map.json"

    faiss.write_index(index, local_index)
    with open(local_idmap, "w") as f:
        json.dump(pids, f)

    size_mb = os.path.getsize(local_index) / 1e6
    print(f"\nIndex saved locally ({size_mb:.1f} MB)")

    if args.smoke:
        print("Smoke test passed. Skipping GCS upload.")
        print(f"  Vectors: {index.ntotal}  Dim: {embeddings.shape[1]}")
        return

    print("Uploading to GCS...")
    _gcs_upload(local_index, FAISS_INDEX_GCS)
    _gcs_upload(local_idmap, FAISS_IDS_GCS)
    print("\nDone. Next: add FAISS retrieval to scripts/06_evaluate.py")


# ── Vertex AI submission ───────────────────────────────────────────────────────

def submit_vertex(args) -> None:
    from google.cloud import aiplatform

    # Load real config values if running locally
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import (
            PROJECT_ID as _PID, BQ_DATASET as _BQ, MODELS_BUCKET as _MB,
            REGION as _R, COLBERT_MODEL_NAME as _CMN, COLBERT_DIM as _CD,
            COLBERT_MAXLEN_DOC as _CMD, COLBERT_CKPT_GCS as _CKPT,
        )
        pid, bq, mb, region = _PID, _BQ, _MB, _R
        cmn, cd, cmd, ckpt  = _CMN, _CD, _CMD, _CKPT
    except ImportError:
        pid, bq, mb, region = PROJECT_ID, BQ_DATASET, MODELS_BUCKET, REGION
        cmn, cd, cmd, ckpt  = COLBERT_MODEL_NAME, COLBERT_DIM, COLBERT_MAXLEN_DOC, COLBERT_CKPT_GCS

    aiplatform.init(project=pid, location=region,
                    staging_bucket=f"gs://{mb}")

    job_args = ["--batch_size", "128"]
    if args.smoke:
        job_args.append("--smoke")

    job = aiplatform.CustomTrainingJob(
        display_name="faiss-index-build" + ("-smoke" if args.smoke else ""),
        script_path="scripts/15_build_faiss_index.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
        requirements=[
            "faiss-cpu==1.7.4",
            "transformers==4.36.0",
            "google-cloud-bigquery",
            "google-cloud-storage",
            "db-dtypes",
            "pandas",
            "pyarrow",
        ],
    )
    print(f"Submitting Vertex AI job (smoke={args.smoke})...")
    job.run(
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        replica_count=1,
        args=job_args,
        environment_variables={
            "GCP_PROJECT_ID":      pid,
            "BQ_DATASET":          bq,
            "GCP_MODELS_BUCKET":   mb,
            "GCP_REGION":          region,
            "COLBERT_MODEL_NAME":  cmn,
            "COLBERT_DIM":         str(cd),
            "COLBERT_MAXLEN_DOC":  str(cmd),
            "COLBERT_CHECKPOINT":  ckpt,
        },
        sync=args.smoke,  # wait for smoke, fire-and-forget for full run
    )
    print(f"Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={pid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit",     action="store_true", help="Submit to Vertex AI")
    parser.add_argument("--smoke",      action="store_true", help="200 products only")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if args.submit:
        submit_vertex(args)
    else:
        main(args)
