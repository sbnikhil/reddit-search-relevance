import torch
import yaml
import os
import pytest
from models.arch.relevance_ranker import RedditRelevanceRanker

with open("config/settings.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cpu")

model_file = f"reddit_ranker_v{cfg['artifacts']['version']}.pt"
model_path = os.path.join(cfg["artifacts"]["model_path"], model_file)


@pytest.fixture
def model():
    model = RedditRelevanceRanker(
        cfg["training"]["model_name"],
        cfg["training"]["extra_feature_dim"],
        dropout=cfg["model_params"].get("dropout_rate", 0.1),
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    return model


def test_model_initialization():
    model = RedditRelevanceRanker(
        cfg["training"]["model_name"],
        cfg["training"]["extra_feature_dim"],
        dropout=cfg["model_params"].get("dropout_rate", 0.1),
    )
    assert model is not None
    assert hasattr(model, "bert")
    assert hasattr(model, "classifier")


def test_model_forward_pass(model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["training"]["model_name"])

    query = "How to fix Python error"
    body = "You can fix this by updating the package"

    inputs = tokenizer(query, body, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    features = torch.tensor([[0.5, 0.5]], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"], features)
        prob = torch.sigmoid(logits)

    assert logits.shape == (1, 1)
    assert 0 <= prob.item() <= 1


def test_model_output_shape(model):
    batch_size = 4
    seq_length = 128

    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones((batch_size, seq_length)).to(device)
    features = torch.rand((batch_size, 2)).to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask, features)

    assert output.shape == (batch_size, 1)
