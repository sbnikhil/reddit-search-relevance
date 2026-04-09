"""Tests for ColBERT architecture."""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.colbert.architecture import ColBERT


@pytest.fixture
def model():
    return ColBERT(model_name="bert-base-uncased", dim=128)


def test_encode_query_shape(model):
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)
    embs = model.encode_query(input_ids, attention_mask)
    assert embs.shape == (2, 32, 128)


def test_encode_document_shape(model):
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128, dtype=torch.long)
    embs = model.encode_document(input_ids, attention_mask)
    assert embs.shape == (2, 128, 128)


def test_maxsim_shape(model):
    query_embs = torch.randn(4, 32, 128)
    query_embs = torch.nn.functional.normalize(query_embs, p=2, dim=-1)
    doc_embs = torch.randn(4, 128, 128)
    doc_embs = torch.nn.functional.normalize(doc_embs, p=2, dim=-1)
    scores = model.maxsim(query_embs, doc_embs)
    assert scores.shape == (4,)


def test_maxsim_scores_bounded(model):
    query_embs = torch.randn(2, 8, 128)
    query_embs = torch.nn.functional.normalize(query_embs, p=2, dim=-1)
    doc_embs = torch.randn(2, 16, 128)
    doc_embs = torch.nn.functional.normalize(doc_embs, p=2, dim=-1)
    scores = model.maxsim(query_embs, doc_embs)
    # MaxSim of normalized vectors: each max sim in [-1,1], summed over Lq=8
    assert scores.max().item() <= 8 + 1e-5
    assert scores.min().item() >= -8 - 1e-5


def test_forward_shape(model):
    query_ids = torch.randint(0, 1000, (2, 32))
    query_mask = torch.ones(2, 32, dtype=torch.long)
    doc_ids = torch.randint(0, 1000, (2, 128))
    doc_mask = torch.ones(2, 128, dtype=torch.long)
    scores = model(query_ids, query_mask, doc_ids, doc_mask)
    assert scores.shape == (2,)


def test_padding_mask_zeros_doc_embs(model):
    input_ids = torch.randint(0, 1000, (1, 128))
    attention_mask = torch.zeros(1, 128, dtype=torch.long)
    embs = model.encode_document(input_ids, attention_mask)
    assert embs.abs().sum().item() == pytest.approx(0.0, abs=1e-5)
