import torch
import yaml
import os
import pandas as pd
from google.cloud import bigquery
from transformers import AutoTokenizer
from models.arch.relevance_ranker import RedditRelevanceRanker

# Load Config
with open("config/settings.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load Model
model = RedditRelevanceRanker(
    cfg["training"]["model_name"],
    cfg["training"]["extra_feature_dim"],
    dropout=cfg["model_params"].get("dropout_rate", 0.1),
).to(device)

model_file = f"reddit_ranker_v{cfg['artifacts']['version']}.pt"
model_path = os.path.join(cfg["artifacts"]["model_path"], model_file)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg["training"]["model_name"])

client = bigquery.Client()
query = """
    SELECT body, expertise_score, utility_score 
    FROM `reddit-search-relevance.reddit_relevance.final_training_set` 
    WHERE expertise_score > 0 OR utility_score > 0
    ORDER BY expertise_score DESC
    LIMIT 10
"""
df = client.query(query).to_dataframe()

# Clean data
df["expertise_score"] = df["expertise_score"].fillna(0.5)
df["utility_score"] = df["utility_score"].fillna(0.5)

print("\nTesting model predictions:")
print("=" * 80)

for idx, row in df.iterrows():
    body = str(row["body"])
    query = " ".join(body.split()[:8])  # Generate query from first 8 words

    # Tokenize
    inputs = tokenizer(
        query,
        body,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg["data_processing"]["max_sequence_length"],
    ).to(device)

    # Prepare features
    features = torch.tensor([[row["expertise_score"], row["utility_score"]]], dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"], features)
        prob = torch.sigmoid(logits).cpu().item()

    print(f"[{prob:.4f}] Exp: {row['expertise_score']:.2f} | Util: {row['utility_score']:.2f}")
    print(f"    {body[:100]}...")
    print()
