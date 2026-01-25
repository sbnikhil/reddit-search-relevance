import pytest
import torch
from models.arch.relevance_ranker import RedditRelevanceRanker

def test_model_initialization():
    model = RedditRelevanceRanker("bert-base-uncased", extra_feature_dim=2, dropout=0.1)
    assert model is not None
    assert model.fc_combined.in_features == 130

def test_model_forward_pass():
    model = RedditRelevanceRanker("bert-base-uncased", extra_feature_dim=2)
    model.eval()
    
    batch_size = 2
    seq_len = 32
    
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    features = torch.randn((batch_size, 2))
    
    with torch.no_grad():
        output = model(input_ids, attention_mask, features)
    
    assert output.shape == (batch_size, 1)

def test_model_output_range():
    model = RedditRelevanceRanker("bert-base-uncased", extra_feature_dim=2)
    model.eval()
    
    input_ids = torch.randint(0, 30522, (1, 32))
    attention_mask = torch.ones((1, 32), dtype=torch.long)
    features = torch.tensor([[0.5, 0.8]])
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask, features)
        prob = torch.sigmoid(logits)
    
    assert 0 <= prob.item() <= 1
