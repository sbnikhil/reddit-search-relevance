"""
Diagnostic script to understand why BERT model outputs near-zero predictions
"""
import torch
import yaml
import os
import numpy as np
from google.cloud import bigquery
from transformers import AutoTokenizer
from models.arch.relevance_ranker import RedditRelevanceRanker

# Load Config
with open("config/settings.yaml", "r") as f:
    cfg = yaml.safe_load(f)

try:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
except:
    device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(cfg['training']['model_name'])
bq_client = bigquery.Client(project=cfg['database']['project_id'])

# Load Model
model = RedditRelevanceRanker(
    cfg['training']['model_name'], 
    cfg['training']['extra_feature_dim'],
    dropout=cfg['model_params'].get('dropout_rate', 0.1)
).to(device)

model_file = f"reddit_ranker_v{cfg['artifacts']['version']}.pt"
model_path = os.path.join(cfg['artifacts']['model_path'], model_file)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("="*80)
print("DIAGNOSIS 1: Check Training Data Distribution")
print("="*80)

# Check label distribution in training data
sql = f"""
SELECT 
    label,
    COUNT(*) as count,
    AVG(expertise_score) as avg_expertise,
    AVG(utility_score) as avg_utility,
    STDDEV(expertise_score) as std_expertise,
    STDDEV(utility_score) as std_utility
FROM `{cfg['database']['source_table']}`
GROUP BY label
"""
df_stats = bq_client.query(sql).to_dataframe()
print("\nTraining Data Statistics:")
print(df_stats.to_string())

total = df_stats['count'].sum()
if len(df_stats) > 1:
    class_0 = df_stats[df_stats['label'] == 0]['count'].values[0] if 0 in df_stats['label'].values else 0
    class_1 = df_stats[df_stats['label'] == 1]['count'].values[0] if 1 in df_stats['label'].values else 0
    imbalance_ratio = max(class_0, class_1) / (min(class_0, class_1) + 1)
    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 10:
        print("WARNING: SEVERE CLASS IMBALANCE - Model will learn to predict majority class!")

print("\n" + "="*80)
print("DIAGNOSIS 2: Test Model on Synthetic Examples")
print("="*80)

# Test 1: Obviously relevant pair
query1 = "How to fix Python import error"
doc1 = "To fix Python import errors, make sure your PYTHONPATH is set correctly and the module is installed. Use pip install to add missing packages."
features1 = torch.tensor([[0.8, 0.9]], dtype=torch.float32).to(device)  # High expertise, high utility

inputs1 = tokenizer(query1, doc1, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
with torch.no_grad():
    logit1 = model(inputs1['input_ids'], inputs1['attention_mask'], features1)
    prob1 = torch.sigmoid(logit1).item()

print(f"\nTest 1 - RELEVANT pair (high features):")
print(f"  Query: {query1}")
print(f"  Doc: {doc1[:80]}...")
print(f"  Features: expertise=0.8, utility=0.9")
print(f"  Logit: {logit1.item():.4f}")
print(f"  Probability: {prob1:.4f}")
print(f"  Expected: >0.5, Got: {prob1:.4f} {'PASS' if prob1 > 0.5 else 'FAIL'}")

# Test 2: Obviously irrelevant pair
query2 = "How to train neural networks"
doc2 = "I love pizza and my favorite topping is pepperoni. Yesterday I went to the movies."
features2 = torch.tensor([[0.1, 0.1]], dtype=torch.float32).to(device)  # Low expertise, low utility

inputs2 = tokenizer(query2, doc2, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
with torch.no_grad():
    logit2 = model(inputs2['input_ids'], inputs2['attention_mask'], features2)
    prob2 = torch.sigmoid(logit2).item()

print(f"\nTest 2 - IRRELEVANT pair (low features):")
print(f"  Query: {query2}")
print(f"  Doc: {doc2[:80]}...")
print(f"  Features: expertise=0.1, utility=0.1")
print(f"  Logit: {logit2.item():.4f}")
print(f"  Probability: {prob2:.4f}")
print(f"  Expected: <0.5, Got: {prob2:.4f} {'PASS' if prob2 < 0.5 else 'FAIL'}")

# Test 3: Default features (what eval likely uses)
query3 = "Python error handling best practices"
doc3 = "When handling errors in Python, use try-except blocks. Catch specific exceptions rather than bare except. Log errors for debugging."
features3 = torch.tensor([[0.5, 0.5]], dtype=torch.float32).to(device)  # Default values

inputs3 = tokenizer(query3, doc3, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
with torch.no_grad():
    logit3 = model(inputs3['input_ids'], inputs3['attention_mask'], features3)
    prob3 = torch.sigmoid(logit3).item()

print(f"\nTest 3 - RELEVANT pair with DEFAULT features (0.5, 0.5):")
print(f"  Query: {query3}")
print(f"  Doc: {doc3[:80]}...")
print(f"  Features: expertise=0.5, utility=0.5")
print(f"  Logit: {logit3.item():.4f}")
print(f"  Probability: {prob3:.4f}")
print(f"  Note: This is what eval uses! Should be >0.3 at least")

print("\n" + "="*80)
print("DIAGNOSIS 3: Check Feature Statistics in Eval Data")
print("="*80)

issues_found = []

sql_eval = f"""
SELECT 
    AVG(expertise_score) as avg_expertise,
    STDDEV(expertise_score) as std_expertise,
    AVG(utility_score) as avg_utility,
    STDDEV(utility_score) as std_utility,
    COUNT(CASE WHEN expertise_score IS NULL THEN 1 END) as null_expertise,
    COUNT(CASE WHEN utility_score IS NULL THEN 1 END) as null_utility
FROM `{cfg['database']['source_table']}`
WHERE label = 1
LIMIT 1000 OFFSET 5000
"""
try:
    df_eval = bq_client.query(sql_eval).to_dataframe()
    if len(df_eval) > 0:
        print("\nEvaluation Set Feature Statistics:")
        print(df_eval.to_string())
        
        if df_eval['std_expertise'].values[0] < 0.1:
            print("WARNING: Expertise scores have almost no variance!")
        if df_eval['std_utility'].values[0] < 0.1:
            print("WARNING: Utility scores have almost no variance!")
    else:
        print("\nNo evaluation data found at offset 5000")
        issues_found.append("No test data available")
except Exception as e:
    print(f"\nCould not fetch eval data: {e}")
    issues_found.append("Evaluation data query failed")

print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

if imbalance_ratio > 10:
    issues_found.append("Severe class imbalance in training data")
if prob1 < 0.5:
    issues_found.append("Model can't predict relevant for obvious relevant pairs")
if prob3 < 0.3:
    issues_found.append("Model outputs near-zero for default features (0.5, 0.5)")

# Check for feature scale mismatch
avg_train_expertise = df_stats['avg_expertise'].mean()
if avg_train_expertise > 10:
    issues_found.append(f"FEATURE SCALE MISMATCH: Training uses expertise~{avg_train_expertise:.0f}, inference uses 0.5!")

if issues_found:
    print("\nISSUES FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print("\nNo obvious issues found - investigate training process")
