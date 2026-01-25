import torch
import torch.nn as nn
import yaml
import os
import random
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from models.arch.relevance_ranker import RedditRelevanceRanker
from sklearn.model_selection import train_test_split

# Load Config
with open("config/settings.yaml", "r") as f:
    cfg = yaml.safe_load(f)

try:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
except:
    device = torch.device("cpu")

print(f"Using device: {device}")


class RedditDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        body = str(row["body"])
        # Simple query proxy for training
        query = " ".join(body.split()[:8])

        encoding = self.tokenizer(
            query, body, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "features": torch.tensor([row["expertise_score"], row["utility_score"]], dtype=torch.float32),
            "label": torch.tensor([row["label"]], dtype=torch.float32),
        }


def train():
    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=cfg["database"]["project_id"])
        sql = f"SELECT body, expertise_score, utility_score, label FROM `{cfg['database']['source_table']}` LIMIT 5000"
        df = client.query(sql).to_dataframe()
        print(f"Loaded {len(df)} samples from BigQuery")
    except Exception as e:
        print(f"BigQuery not available, using local sample data")

        import json

        sample_path = "data/sample_posts.json"
        if os.path.exists(sample_path):
            with open(sample_path, "r") as f:
                posts = json.load(f)
            df = pd.DataFrame(posts)
            print(f"Loaded {len(df)} samples from {sample_path}")
        else:
            print("ERROR: No data source available. Run: make create-samples")
            exit(1)

    df = df.dropna(subset=["body", "label"])

    print(f"Original feature ranges (before fillna):")
    print(f"  Expertise: [{df['expertise_score'].min():.2f}, {df['expertise_score'].max():.2f}]")
    print(f"  Utility: [{df['utility_score'].min():.2f}, {df['utility_score'].max():.2f}]")

    exp_min, exp_max = df["expertise_score"].min(), df["expertise_score"].max()
    util_min, util_max = df["utility_score"].min(), df["utility_score"].max()

    if pd.isna(exp_min) or pd.isna(exp_max) or exp_min == exp_max:
        exp_min, exp_max = 0.0, 1000.0
    if pd.isna(util_min) or pd.isna(util_max) or util_min == util_max:
        util_min, util_max = 0.0, 2.0

    df["expertise_score"] = df["expertise_score"].fillna(
        df["expertise_score"].median() if not pd.isna(df["expertise_score"].median()) else 0.0
    )
    df["utility_score"] = df["utility_score"].fillna(
        df["utility_score"].median() if not pd.isna(df["utility_score"].median()) else 0.0
    )

    df["expertise_score"] = (df["expertise_score"] - exp_min) / (exp_max - exp_min + 1e-10)
    df["utility_score"] = (df["utility_score"] - util_min) / (util_max - util_min + 1e-10)

    print(
        f"Normalized ranges - Expertise: [{df['expertise_score'].min():.2f}, {df['expertise_score'].max():.2f}], Utility: [{df['utility_score'].min():.2f}, {df['utility_score'].max():.2f}]"
    )

    import json

    norm_params = {
        "expertise_min": float(exp_min),
        "expertise_max": float(exp_max),
        "utility_min": float(util_min),
        "utility_max": float(util_max),
    }
    os.makedirs(cfg["artifacts"]["model_path"], exist_ok=True)
    with open(os.path.join(cfg["artifacts"]["model_path"], "feature_norm_params.json"), "w") as f:
        json.dump(norm_params, f, indent=2)

    print(f"Cleaned samples: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=cfg["training"]["val_split"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["training"]["model_name"])

    train_loader = DataLoader(
        RedditDataset(train_df, tokenizer, cfg["data_processing"]["max_sequence_length"]),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        RedditDataset(val_df, tokenizer, cfg["data_processing"]["max_sequence_length"]),
        batch_size=cfg["training"]["batch_size"],
    )

    model = RedditRelevanceRanker(
        cfg["training"]["model_name"], cfg["training"]["extra_feature_dim"], dropout=cfg["model_params"]["dropout_rate"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    model_save_name = f"reddit_ranker_v{cfg['artifacts']['version']}.pt"
    save_path = os.path.join(cfg["artifacts"]["model_path"], model_save_name)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["features"].to(device)
            )
            loss = criterion(outputs, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["features"].to(device)
                )
                total_val_loss += criterion(outputs, batch["label"].to(device)).item()

        avg_val = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs(cfg["artifacts"]["model_path"], exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")


if __name__ == "__main__":
    train()
