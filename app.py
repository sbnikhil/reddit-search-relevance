import pysolr
import torch
import yaml
import os
import json
import numpy as np
from transformers import AutoTokenizer
from models.arch.relevance_ranker import RedditRelevanceRanker

# Load Config
with open("config/settings.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Load feature normalization parameters
norm_params_path = os.path.join(cfg['artifacts']['model_path'], 'feature_norm_params.json')
if os.path.exists(norm_params_path):
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    print("Loaded feature normalization params")
else:
    print("Warning: No normalization params found, using defaults")
    norm_params = {'expertise_min': 0, 'expertise_max': 1, 'utility_min': 0, 'utility_max': 1}

try:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
except:
    device = torch.device("cpu")

# Initialize Solr
solr = pysolr.Solr(cfg['search_tuning']['solr_url'], timeout=10)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg['training']['model_name'])

# Load Model
model = RedditRelevanceRanker(
    cfg['training']['model_name'], 
    cfg['training']['extra_feature_dim'],
    dropout=cfg['model_params'].get('dropout_rate', 0.1)
).to(device)

model_file = f"reddit_ranker_v{cfg['artifacts']['version']}.pt"
model_path = os.path.join(cfg['artifacts']['model_path'], model_file)

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model: {model_file}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
else:
    print(f"Error: Model not found at {model_path}")
    print(f"Run 'make train' to create the model first")
    exit(1)

model.eval()

def search(query_text, top_k=10):
    """
    Search Reddit posts and re-rank using trained BERT model
    """
    # Stage 1: Solr Retrieval
    try:
        results = solr.search(f"body_t:({query_text})", rows=cfg['search_tuning']['k_recall'], fl="*,score")
    except Exception as e:
        print(f"Solr error: {e}")
        return
    
    if not results or len(results) == 0:
        print("No results found.")
        return
    
    # Stage 2: BERT Re-ranking
    batch_docs = [doc.get('body_t', '') for doc in results]
    
    inputs = tokenizer(
        [query_text] * len(results), 
        batch_docs, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=cfg['data_processing']['max_sequence_length']
    ).to(device)
    
    # Extract features (use defaults if missing) and normalize
    raw_features = [[float(doc.get('expertise_score', 0.0)), float(doc.get('utility_score', 0.0))] for doc in results]
    
    # Normalize using training-time scale
    normalized_features = []
    for exp, util in raw_features:
        exp_norm = (exp - norm_params['expertise_min']) / (norm_params['expertise_max'] - norm_params['expertise_min'] + 1e-6)
        util_norm = (util - norm_params['utility_min']) / (norm_params['utility_max'] - norm_params['utility_min'] + 1e-6)
        exp_norm = max(0.0, min(1.0, exp_norm))  # Clip to [0, 1]
        util_norm = max(0.0, min(1.0, util_norm))
        normalized_features.append([exp_norm, util_norm])
    
    features = torch.tensor(normalized_features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'], features)
        bert_scores = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Stage 3: Hybrid Scoring
    solr_scores = np.array([float(doc.get('score', 0.0)) for doc in results])
    
    # Normalize Solr scores to [0, 1]
    s_min, s_max = solr_scores.min(), solr_scores.max()
    solr_norm = (solr_scores - s_min) / (s_max - s_min + 1e-6)
    
    # Combine scores
    alpha = cfg['search_tuning']['alpha']
    hybrid_scores = (alpha * solr_norm) + ((1 - alpha) * bert_scores)
    
    # Sort by hybrid score
    ranked_results = sorted(
        zip(hybrid_scores, bert_scores, solr_norm, batch_docs), 
        key=lambda x: x[0], 
        reverse=True
    )
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Search results for: '{query_text}'")
    print(f"{'='*80}\n")
    
    for i, (h_score, b_score, s_score, body) in enumerate(ranked_results[:top_k], 1):
        print(f"[{i}] Hybrid: {h_score:.4f} | BERT: {b_score:.4f} | Solr: {s_score:.4f}")
        print(f"    {body[:150]}...")
        print()

if __name__ == "__main__":
    query = input("Search Reddit: ")
    search(query)