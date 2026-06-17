"""
Train cross-encoder with ρ-conditioned rationale distillation.

Architecture: ModernBERT-base → 4-class head (E/S/C/I)
Score:        P(E) + 0.1·P(S) + 0.01·P(C)
Loss:         (1 - λ(ρ)) · CE(logits, hard_label)
            +      λ(ρ)  · KL(softmax(logits/T) ‖ LLM_soft_label)
Where:        λ(ρ) = clip(1 - ρ, 0, 1)
              ρ = Spearman(ColBERT scores, ESCI gain) per training query

Self-contained: all model code, config, and GCS helpers are inlined so this
script can run on Vertex AI without the full project package.

Training data (from BigQuery):
  examples table    → query, product_id, esci_label, split='train'
  products table    → product_title, product_description
  colbert_scores    → used to compute per-query ρ
  rationale_labels  → LLM soft label distributions (from script 16)

Usage:
  python scripts/17_train_cross_encoder.py --smoke     # 100 pairs, local validation
  python scripts/17_train_cross_encoder.py --submit    # Vertex AI full training
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.cloud import bigquery, storage
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ID    = os.environ.get("GCP_PROJECT_ID",     "search-relevance-500100")
BQ_DATASET    = os.environ.get("BQ_DATASET",         "esci_search")
MODELS_BUCKET = os.environ.get("GCP_MODELS_BUCKET",  "search-models-nikhil")
REGION        = os.environ.get("GCP_REGION",         "us-central1")
CE_MODEL_NAME = os.environ.get("CE_MODEL_NAME",      "answerdotai/ModernBERT-base")
CE_MAXLEN     = int(os.environ.get("CE_MAXLEN",      "256"))
CE_BATCH_SIZE = int(os.environ.get("CE_BATCH_SIZE",  "32"))
CE_EPOCHS     = int(os.environ.get("CE_EPOCHS",      "3"))
CE_LR         = float(os.environ.get("CE_LR",        "2e-5"))
CE_WARMUP     = int(os.environ.get("CE_WARMUP",      "500"))
KL_TEMP       = float(os.environ.get("KL_TEMP",      "2.0"))

CKPT_GCS_PREFIX = f"gs://{MODELS_BUCKET}/cross_encoder"
LABEL_TO_IDX    = {"E": 0, "S": 1, "C": 2, "I": 3}


# ── Inlined model ───────────────────────────────────────────────────────────────

class CrossEncoder(nn.Module):
    SCORE_WEIGHTS = torch.tensor([1.0, 0.1, 0.01, 0.0])

    def __init__(self, model_name: str = CE_MODEL_NAME):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

    @torch.no_grad()
    def score(self, input_ids, attention_mask):
        probs   = torch.softmax(self.forward(input_ids, attention_mask), dim=-1)
        weights = self.SCORE_WEIGHTS.to(probs.device)
        return (probs * weights).sum(dim=-1)


# ── Inlined loss ────────────────────────────────────────────────────────────────

class RhoConditionedLoss(nn.Module):
    def __init__(self, temperature: float = KL_TEMP):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, hard_labels, soft_labels, rho, has_soft):
        ce_loss  = self.ce(logits, hard_labels)
        log_prob = F.log_softmax(logits / self.temperature, dim=-1)
        kl_loss  = F.kl_div(log_prob, soft_labels.clamp(min=1e-8),
                             reduction="none").sum(dim=-1)
        lam      = (1.0 - rho).clamp(0.0, 1.0) * has_soft.float()
        return ((1.0 - lam) * ce_loss + lam * kl_loss).mean()


# ── GCS helpers ─────────────────────────────────────────────────────────────────

def _gcs_upload(local_path: str, gcs_uri: str) -> None:
    bucket, blob = gcs_uri.replace("gs://", "").split("/", 1)
    storage.Client().bucket(bucket).blob(blob).upload_from_filename(local_path)
    print(f"  Uploaded → {gcs_uri}")


# ── Data loading ────────────────────────────────────────────────────────────────

def load_training_data(smoke: bool) -> pd.DataFrame:
    bq    = bigquery.Client(project=PROJECT_ID)
    limit = "LIMIT 400" if smoke else ""

    sql = f"""
    WITH ranked AS (
        SELECT
            CAST(e.query_id AS STRING) AS query_id,
            RANK() OVER (PARTITION BY e.query_id ORDER BY c.colbert_score DESC) AS colbert_rank,
            RANK() OVER (PARTITION BY e.query_id ORDER BY e.gain        DESC) AS gain_rank
        FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
        JOIN `{PROJECT_ID}.{BQ_DATASET}.colbert_scores` c
            ON CAST(e.query_id AS STRING) = c.query_id
            AND CAST(e.product_id AS STRING) = c.product_id
        WHERE e.split = 'train'
    ),
    query_rho AS (
        SELECT query_id, CORR(colbert_rank, gain_rank) AS rho
        FROM ranked
        GROUP BY query_id
    )
    SELECT
        CAST(e.query_id AS STRING)   AS query_id,
        CAST(e.product_id AS STRING) AS product_id,
        e.query,
        e.esci_label,
        COALESCE(p.product_title, '')       AS title,
        COALESCE(p.product_description, '') AS description,
        COALESCE(r.rho, 0.5)               AS rho,
        COALESCE(rl.p_exact,      -1.0)    AS p_exact,
        COALESCE(rl.p_substitute, -1.0)    AS p_substitute,
        COALESCE(rl.p_complement, -1.0)    AS p_complement,
        COALESCE(rl.p_irrelevant, -1.0)    AS p_irrelevant
    FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
    JOIN `{PROJECT_ID}.{BQ_DATASET}.products` p USING (product_id)
    LEFT JOIN query_rho r ON CAST(e.query_id AS STRING) = r.query_id
    LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.rationale_labels` rl
        ON CAST(e.query_id AS STRING) = rl.query_id
        AND CAST(e.product_id AS STRING) = rl.product_id
    WHERE e.split = 'train'
    ORDER BY rho ASC
    {limit}
    """
    print("Loading training data from BigQuery...")
    df = bq.query(sql).to_dataframe()
    # has_soft: True when LLM soft label exists (p_exact != -1)
    df["rho"]     = df["rho"].fillna(0.5)   # CORR() returns NULL for single-product queries
    df["has_soft"] = df["p_exact"] >= 0
    # replace sentinel with uniform distribution where no soft label
    for col in ["p_exact", "p_substitute", "p_complement", "p_irrelevant"]:
        df[col] = df[col].clip(lower=0.0)
    print(f"  {len(df):,} pairs  has_soft={df['has_soft'].sum():,}")
    return df


# ── Dataset ─────────────────────────────────────────────────────────────────────

class ESCIDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row    = self.df.iloc[idx]
        query  = str(row["query"])
        doc    = f"{row['title']} {row['description']}"
        enc    = self.tokenizer(
            query, doc,
            max_length=CE_MAXLEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "hard_label":     torch.tensor(LABEL_TO_IDX[row["esci_label"]], dtype=torch.long),
            "soft_label":     torch.tensor(
                [row["p_exact"], row["p_substitute"],
                 row["p_complement"], row["p_irrelevant"]], dtype=torch.float32
            ),
            "rho":      torch.tensor(float(row["rho"]),      dtype=torch.float32),
            "has_soft": torch.tensor(bool(row["has_soft"]),  dtype=torch.bool),
        }


# ── Training ────────────────────────────────────────────────────────────────────

def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df        = load_training_data(smoke=args.smoke)
    tokenizer = AutoTokenizer.from_pretrained(CE_MODEL_NAME)
    dataset   = ESCIDataset(df, tokenizer)
    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=CE_BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset):,} pairs, {len(loader)} steps/epoch")

    model     = CrossEncoder(model_name=CE_MODEL_NAME).to(device)
    criterion = RhoConditionedLoss(temperature=KL_TEMP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CE_LR, weight_decay=0.01)
    total_steps = len(loader) * CE_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=CE_WARMUP, num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, CE_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(loader):
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            hard_label = batch["hard_label"].to(device)
            soft_label = batch["soft_label"].to(device)
            rho        = batch["rho"].to(device)
            has_soft   = batch["has_soft"].to(device)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(input_ids, attn_mask)
                loss   = criterion(logits, hard_label, soft_label, rho, has_soft)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch}  step {step}/{len(loader)}"
                      f"  loss={loss.item():.4f}  {elapsed:.0f}s")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} complete. avg_loss={avg_loss:.4f}")

        if not args.smoke:
            ckpt_local = f"/tmp/cross_encoder_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_local)
            _gcs_upload(ckpt_local, f"{CKPT_GCS_PREFIX}/epoch_{epoch}/model.pt")

    if args.smoke:
        print("\nSmoke test passed. No checkpoint saved.")
    else:
        final_local = "/tmp/cross_encoder_final.pt"
        torch.save(model.state_dict(), final_local)
        _gcs_upload(final_local, f"{CKPT_GCS_PREFIX}/final/model.pt")
        # Save model name so evaluate can load it
        meta = {"model_name": CE_MODEL_NAME, "maxlen": CE_MAXLEN, "epochs": CE_EPOCHS}
        with open("/tmp/cross_encoder_meta.json", "w") as f:
            json.dump(meta, f)
        _gcs_upload("/tmp/cross_encoder_meta.json",
                    f"{CKPT_GCS_PREFIX}/final/meta.json")
        print(f"\nTraining done. Checkpoint: {CKPT_GCS_PREFIX}/final/model.pt")
        print("Next: add cross-encoder scoring to scripts/06_evaluate.py")


# ── Vertex AI submission ────────────────────────────────────────────────────────

def submit_vertex(args) -> None:
    from google.cloud import aiplatform

    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import (
            PROJECT_ID as _PID, BQ_DATASET as _BQ, MODELS_BUCKET as _MB, REGION as _R,
        )
        pid, bq, mb, region = _PID, _BQ, _MB, _R
    except ImportError:
        pid, bq, mb, region = PROJECT_ID, BQ_DATASET, MODELS_BUCKET, REGION

    aiplatform.init(project=pid, location=region, staging_bucket=f"gs://{mb}")

    job_args = []
    if args.smoke:
        job_args.append("--smoke")

    job = aiplatform.CustomTrainingJob(
        display_name="cross-encoder-rho-distillation" + ("-smoke" if args.smoke else ""),
        script_path="scripts/17_train_cross_encoder.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
        requirements=[
            "transformers>=4.48.0",
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
            "GCP_PROJECT_ID":   pid,
            "BQ_DATASET":       bq,
            "GCP_MODELS_BUCKET": mb,
            "GCP_REGION":       region,
            "CE_MODEL_NAME":    CE_MODEL_NAME,
            "CE_MAXLEN":        str(CE_MAXLEN),
            "CE_BATCH_SIZE":    str(CE_BATCH_SIZE),
            "CE_EPOCHS":        str(CE_EPOCHS),
            "CE_LR":            str(CE_LR),
            "CE_WARMUP":        str(CE_WARMUP),
        },
        sync=args.smoke,
    )
    print(f"Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={pid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--smoke",  action="store_true",
                        help="100 pairs, validate pipeline only, no GCS upload")
    args = parser.parse_args()

    if args.submit:
        submit_vertex(args)
    else:
        train(args)
