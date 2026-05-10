import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROJECT_ID, MODELS_BUCKET, DATA_BUCKET, REGION,
    COLBERT_DIM, COLBERT_MODEL_NAME, COLBERT_WARMUP_STEPS,
    COLBERT_BATCH_SIZE, COLBERT_EPOCHS, COLBERT_LR,
)
from models.colbert.architecture import ColBERT
from models.colbert.dataset import ESCITripletDataset
from models.colbert.losses import colbert_loss
from utils.gcs import download, upload
from google.cloud import aiplatform

TRIPLETS_GCS   = f"gs://{DATA_BUCKET}/esci/triplets/train_triplets.parquet"
CHECKPOINT_DIR = f"gs://{MODELS_BUCKET}/colbert/"


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer, get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Downloading triplets...")
    download(TRIPLETS_GCS, "/tmp/train_triplets.parquet")

    tokenizer = BertTokenizer.from_pretrained(COLBERT_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})

    dataset = ESCITripletDataset("/tmp/train_triplets.parquet", tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Dataset: {len(dataset)} triplets, {len(loader)} steps/epoch")

    model = ColBERT(dim=COLBERT_DIM).to(device)
    model.bert.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=COLBERT_WARMUP_STEPS, num_training_steps=total_steps
    )

    aiplatform.init(project=PROJECT_ID, location=REGION, experiment="colbert-esci")
    run_suffix = int(time.time())

    start_epoch = 1
    for e in range(args.epochs, 0, -1):
        local_ckpt = f"/tmp/colbert_epoch_{e}_resume.pt"
        try:
            download(f"{CHECKPOINT_DIR}epoch_{e}/model.pt", local_ckpt)
            model.load_state_dict(torch.load(local_ckpt, map_location=device))
            start_epoch = e + 1
            print(f"Resumed from epoch {e} checkpoint.")
            break
        except Exception:
            continue

    if start_epoch > args.epochs:
        print("All epochs already completed.")
        return

    print(f"Starting from epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            q_ids, q_mask, p_ids, p_mask, n_ids, n_mask = [b.to(device) for b in batch]
            q_embs = model.encode_query(q_ids, q_mask)
            p_embs = model.encode_document(p_ids, p_mask)
            n_embs = model.encode_document(n_ids, n_mask)
            loss = colbert_loss(q_embs, p_embs, n_embs, model)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step}/{len(loader)} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        try:
            with aiplatform.start_run(run=f"colbert-epoch-{epoch}-{run_suffix}"):
                aiplatform.log_metrics({"loss": avg_loss, "epoch": epoch, "batch_size": args.batch_size})
        except Exception as log_err:
            print(f"Metric logging skipped: {log_err}")

        ckpt = f"/tmp/colbert_epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt)
        upload(ckpt, f"{CHECKPOINT_DIR}epoch_{epoch}/model.pt")
        print(f"Checkpoint saved: epoch {epoch}")

    final = "/tmp/colbert_final.pt"
    torch.save(model.state_dict(), final)
    upload(final, f"{CHECKPOINT_DIR}final/model.pt")
    print("Training complete.")


def submit_vertex_job(args):
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{MODELS_BUCKET}",
    )
    job = aiplatform.CustomTrainingJob(
        display_name="colbert-esci-training",
        script_path="scripts/04_train_colbert.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
        requirements=[
            "transformers==4.36.0", "rank_bm25", "google-cloud-bigquery",
            "google-cloud-storage", "google-cloud-aiplatform", "db-dtypes", "pandas", "pyarrow",
        ],
    )
    job.run(
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        replica_count=1,
        args=["--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--lr", str(args.lr)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=COLBERT_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=COLBERT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=COLBERT_LR)
    parser.add_argument("--submit",     action="store_true")
    args = parser.parse_args()

    if args.submit:
        submit_vertex_job(args)
    else:
        train(args)
