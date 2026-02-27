import argparse
import os
import time

from google.cloud import storage, aiplatform

PROJECT_ID = "reddit-search-relevance-485717"
TRIPLETS_GCS = "gs://reddit-search-relevance-data/esci/triplets/train_triplets.parquet"
CHECKPOINT_DIR = "gs://reddit-search-relevance-models/colbert/"

def build_model(model_name="bert-base-uncased", dim=128):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertModel

    class ColBERT(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained(model_name)
            self.linear = nn.Linear(self.bert.config.hidden_size, dim)
            self.dim = dim

        def encode_query(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embs = self.linear(out.last_hidden_state)
            return F.normalize(embs, p=2, dim=-1)

        def encode_document(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embs = self.linear(out.last_hidden_state)
            embs = F.normalize(embs, p=2, dim=-1)
            mask = attention_mask.unsqueeze(-1).float()
            return embs * mask

        def maxsim(self, query_embs, doc_embs):
            sim = torch.bmm(query_embs, doc_embs.transpose(1, 2))
            return sim.max(dim=2).values.sum(dim=1)

        def forward(self, query_ids, query_mask, doc_ids, doc_mask):
            return self.maxsim(
                self.encode_query(query_ids, query_mask),
                self.encode_document(doc_ids, doc_mask),
            )

    return ColBERT()

def build_dataset(parquet_path, tokenizer=None, model_name="bert-base-uncased", query_maxlen=32, doc_maxlen=128):
    import pandas as pd
    import torch
    from torch.utils.data import Dataset
    from transformers import BertTokenizer

    class ESCITripletDataset(Dataset):
        def __init__(self):
            self.df = pd.read_parquet(parquet_path)
            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})

        def _tok(self, text, prefix, maxlen):
            enc = self.tokenizer(
                f"{prefix} {text}",
                max_length=maxlen,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            q_ids, q_mask = self._tok(row["query"], "[Q]", query_maxlen)
            pos_text = row["positive_title"] + " " + (row["positive_description"] or "")
            neg_text = row["negative_title"] + " " + (row["negative_description"] or "")
            p_ids, p_mask = self._tok(pos_text, "[D]", doc_maxlen)
            n_ids, n_mask = self._tok(neg_text, "[D]", doc_maxlen)
            return q_ids, q_mask, p_ids, p_mask, n_ids, n_mask

    return ESCITripletDataset()

def colbert_loss(query_embs, pos_embs, neg_embs, model, temperature=0.05):
    import torch
    import torch.nn.functional as F
    B = query_embs.size(0)
    neg_scores = model.maxsim(query_embs, neg_embs)
    sim = torch.einsum('iqd,jld->ijql', query_embs, pos_embs)
    scores_matrix = sim.max(dim=3).values.sum(dim=2)
    logits = torch.cat([scores_matrix, neg_scores.unsqueeze(1)], dim=1) / temperature
    labels = torch.arange(B, device=query_embs.device)
    return F.cross_entropy(logits, labels)

def download_from_gcs(gcs_path, local_path):
    bucket = gcs_path.replace("gs://", "").split("/")[0]
    blob = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
    storage.Client(project=PROJECT_ID).bucket(bucket).blob(blob).download_to_filename(local_path)

def upload_to_gcs(local_path, gcs_path):
    bucket = gcs_path.replace("gs://", "").split("/")[0]
    blob = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
    storage.Client(project=PROJECT_ID).bucket(bucket).blob(blob).upload_from_filename(local_path)

def train(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from transformers import BertTokenizer

    print("Downloading triplets...")
    download_from_gcs(TRIPLETS_GCS, "/tmp/train_triplets.parquet")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})

    dataset = build_dataset("/tmp/train_triplets.parquet", tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Dataset: {len(dataset)} triplets, {len(loader)} steps/epoch")

    model = build_model().to(device)
    model.bert.resize_token_embeddings(len(tokenizer))  # expand embeddings for [Q] and [D]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

    aiplatform.init(project=PROJECT_ID, location="us-central1", experiment="colbert-esci")
    run_suffix = int(time.time())

    start_epoch = 1
    for e in range(args.epochs, 0, -1):
        gcs_ckpt = f"{CHECKPOINT_DIR}epoch_{e}/model.pt"
        local_ckpt = f"/tmp/colbert_epoch_{e}_resume.pt"
        try:
            download_from_gcs(gcs_ckpt, local_ckpt)
            model.load_state_dict(torch.load(local_ckpt, map_location=device))
            start_epoch = e + 1
            print(f"Resumed from epoch {e} checkpoint.")
            break
        except Exception:
            continue

    if start_epoch > args.epochs:
        print("All epochs already completed.")
    else:
        print(f"Starting from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            query_ids, query_mask, pos_ids, pos_mask, neg_ids, neg_mask = [b.to(device) for b in batch]
            query_embs = model.encode_query(query_ids, query_mask)
            pos_embs = model.encode_document(pos_ids, pos_mask)
            neg_embs = model.encode_document(neg_ids, neg_mask)
            loss = colbert_loss(query_embs, pos_embs, neg_embs, model)
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
        upload_to_gcs(ckpt, f"{CHECKPOINT_DIR}epoch_{epoch}/model.pt")
        print(f"Checkpoint saved: epoch {epoch}")

    final = "/tmp/colbert_final.pt"
    torch.save(model.state_dict(), final)
    upload_to_gcs(final, f"{CHECKPOINT_DIR}final/model.pt")
    print("Training complete.")

def submit_vertex_job(args):
    aiplatform.init(
        project=PROJECT_ID,
        location="us-central1",
        staging_bucket="gs://reddit-search-relevance-models",
    )
    job = aiplatform.CustomTrainingJob(
        display_name="colbert-esci-training",
        script_path="scripts/04_train_colbert.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
        requirements=["transformers==4.36.0", "rank_bm25", "google-cloud-bigquery",
                      "google-cloud-storage", "google-cloud-aiplatform", "db-dtypes", "pandas", "pyarrow"],
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.submit:
        submit_vertex_job(args)
    else:
        train(args)
