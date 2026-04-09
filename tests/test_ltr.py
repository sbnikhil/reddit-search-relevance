"""Tests for LambdaMART LTR model."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ltr.lambdamart import (
    LambdaMARTRanker,
    build_features,
    gain_to_label,
    FEATURE_NAMES,
)


def make_dummy_dataset(n_queries=20, products_per_query=5):
    features, labels, groups = [], [], []
    rng = np.random.default_rng(42)
    for _ in range(n_queries):
        for _ in range(products_per_query):
            features.append(rng.random(len(FEATURE_NAMES)).tolist())
            labels.append(int(rng.choice([0, 1, 2, 3])))
        groups.append(products_per_query)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32), groups


# ── gain_to_label ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gain,expected", [
    (1.0, 3),
    (0.1, 2),
    (0.01, 1),
    (0.0, 0),
    (0.005, 0),   # below C threshold
    (0.15, 2),    # above S threshold
])
def test_gain_to_label(gain, expected):
    assert gain_to_label(gain) == expected


# ── build_features ────────────────────────────────────────────────────────────

def test_build_features_length():
    f = build_features("red shoes", "Red Running Shoes", "Great shoes", "Nike")
    assert len(f) == len(FEATURE_NAMES)


def test_build_features_scores_passthrough():
    f = build_features("red shoes", "title", "desc", "brand",
                       bm25_score=0.9, colbert_score=0.7)
    assert f[FEATURE_NAMES.index("bm25_score")] == pytest.approx(0.9)
    assert f[FEATURE_NAMES.index("colbert_score")] == pytest.approx(0.7)


def test_build_features_full_title_overlap():
    # query tokens are a subset of title tokens → overlap = 1.0
    f = build_features("red shoes", "red shoes extra", "desc", "brand")
    assert f[FEATURE_NAMES.index("title_query_overlap")] == pytest.approx(1.0)


def test_build_features_no_overlap():
    f = build_features("laptop", "running shoes", "comfortable", "Nike")
    assert f[FEATURE_NAMES.index("title_query_overlap")] == pytest.approx(0.0)


def test_build_features_brand_match():
    f = build_features("nike shoes", "some product", "desc", "Nike")
    assert f[FEATURE_NAMES.index("brand_match")] == pytest.approx(1.0)


def test_build_features_no_brand_match():
    f = build_features("adidas shoes", "some product", "desc", "Nike")
    assert f[FEATURE_NAMES.index("brand_match")] == pytest.approx(0.0)


def test_build_features_has_description():
    with_desc = build_features("q", "t", "some description", "b")
    no_desc = build_features("q", "t", "", "b")
    assert with_desc[FEATURE_NAMES.index("has_description")] == 1.0
    assert no_desc[FEATURE_NAMES.index("has_description")] == 0.0


def test_build_features_none_inputs():
    # Should not raise on None fields
    f = build_features("query", None, None, None)
    assert len(f) == len(FEATURE_NAMES)


# ── LambdaMARTRanker ──────────────────────────────────────────────────────────

def test_train_and_predict():
    ranker = LambdaMARTRanker()
    X, y, groups = make_dummy_dataset()
    ranker.train(X, y, groups)
    preds = ranker.predict(X[:5])
    assert preds.shape == (5,)
    assert np.isfinite(preds).all()


def test_predict_before_train_raises():
    ranker = LambdaMARTRanker()
    with pytest.raises(RuntimeError, match="not trained"):
        ranker.predict(np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32))


def test_feature_importance_before_train_raises():
    ranker = LambdaMARTRanker()
    with pytest.raises(RuntimeError, match="not trained"):
        ranker.feature_importance()


def test_feature_importance_keys():
    ranker = LambdaMARTRanker()
    X, y, groups = make_dummy_dataset()
    ranker.train(X, y, groups)
    fi = ranker.feature_importance()
    assert set(fi.keys()) == set(FEATURE_NAMES)


def test_train_with_eval_set():
    ranker = LambdaMARTRanker()
    X, y, groups = make_dummy_dataset(n_queries=30)
    split = 20 * 5  # first 20 queries for train
    X_train, y_train, g_train = X[:split], y[:split], groups[:20]
    X_val, y_val, g_val = X[split:], y[split:], groups[20:]
    ranker.train(X_train, y_train, g_train,
                 eval_features=X_val, eval_labels=y_val, eval_groups=g_val,
                 num_boost_round=50, early_stopping_rounds=10)
    assert ranker.model is not None


def test_groups_sum_equals_features():
    X, y, groups = make_dummy_dataset(n_queries=5, products_per_query=4)
    assert sum(groups) == len(X) == len(y)


def test_predict_order_consistent():
    ranker = LambdaMARTRanker()
    X, y, groups = make_dummy_dataset()
    ranker.train(X, y, groups)
    p1 = ranker.predict(X)
    p2 = ranker.predict(X)
    np.testing.assert_array_equal(p1, p2)
