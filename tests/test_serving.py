"""Tests for the FastAPI serving endpoint.

Search tests inject a minimal mock pipeline into _state so they run without
GCS access, BigQuery, or GPU. Health and metrics tests need no mocks.
"""
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import serving.app as app_module
from serving.app import app


# ── Mock pipeline components ──────────────────────────────────────────────────

class _MockBM25:
    """Returns uniform scores for any query."""
    def get_scores(self, tokens):
        return np.ones(len(_MOCK_PRODUCT_IDS))


class _MockColBERT:
    """Returns small random embeddings; deterministic via fixed seed."""
    def encode_query(self, input_ids, attention_mask):
        torch.manual_seed(0)
        return torch.randn(input_ids.size(0), 4, 128)  # (B, Lq, dim)

    def encode_document(self, input_ids, attention_mask):
        torch.manual_seed(1)
        return torch.randn(input_ids.size(0), 8, 128)  # (B, Ld, dim)


class _MockTokenizer:
    """Returns minimal tensors shaped as (B, seq_len)."""
    def __call__(self, texts, padding=True, truncation=True, max_length=32, return_tensors="pt"):
        B = len(texts) if isinstance(texts, list) else 1
        return {"input_ids": torch.zeros(B, 4, dtype=torch.long),
                "attention_mask": torch.ones(B, 4, dtype=torch.long)}


class _MockLTR:
    """Returns decreasing scores so results have a deterministic order."""
    def predict(self, features):
        return np.linspace(1.0, 0.0, num=len(features))


_MOCK_PRODUCT_IDS = [f"pid_{i}" for i in range(10)]
_MOCK_CATALOG = {
    pid: {"title": f"Product {i}", "description": "A nice product.", "brand": "BrandX"}
    for i, pid in enumerate(_MOCK_PRODUCT_IDS)
}

_MOCK_STATE = {
    "device":      torch.device("cpu"),
    "product_ids": _MOCK_PRODUCT_IDS,
    "bm25":        _MockBM25(),
    "catalog":     _MOCK_CATALOG,
    "colbert":     _MockColBERT(),
    "tokenizer":   _MockTokenizer(),
    "ltr":         _MockLTR(),
}


@pytest.fixture(autouse=True)
def inject_mock_state():
    """Populate _state with mock pipeline before each test, then clean up."""
    app_module._state.update(_MOCK_STATE)
    yield
    app_module._state.clear()


client = TestClient(app)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert data["ready"] is True


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "ndcg_10" in data
    assert "mrr_10" in data
    assert "p99_latency_ms" in data


def test_search_returns_200():
    r = client.post("/search", json={"query": "red running shoes", "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert data["query"] == "red running shoes"
    assert "latency_ms" in data


def test_search_default_top_k():
    r = client.post("/search", json={"query": "laptop bag"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) <= 10


def test_search_top_k_respected():
    r = client.post("/search", json={"query": "coffee maker", "top_k": 3})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3


def test_search_response_schema():
    r = client.post("/search", json={"query": "coffee maker", "top_k": 5})
    assert r.status_code == 200
    for result in r.json()["results"]:
        assert "product_id" in result
        assert "title" in result
        assert "score" in result
        assert "stage_scores" in result
        ss = result["stage_scores"]
        assert "bm25" in ss
        assert "colbert" in ss
        assert "ltr" in ss


def test_search_503_when_not_ready(monkeypatch):
    monkeypatch.setattr(app_module, "_state", {})
    r = client.post("/search", json={"query": "anything"})
    assert r.status_code == 503
