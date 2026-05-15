import numpy as np
import lightgbm as lgb
from config import LAMBDAMART_PARAMS, LTR_NUM_BOOST_ROUND, LTR_EARLY_STOPPING_ROUNDS


def bm25_scores_for_group(query: str, titles: list, descriptions: list) -> np.ndarray:
    """BM25 scores for a single query's candidate set, normalized 0-1."""
    from rank_bm25 import BM25Okapi
    corpus = [
        ((t or "") + " " + (d or "")).lower().split() or [""]
        for t, d in zip(titles, descriptions)
    ]
    scores = BM25Okapi(corpus).get_scores(query.lower().split())
    max_s = scores.max()
    return scores / max_s if max_s > 0 else scores


# ESCI float gain -> integer relevance level for lambdarank (I=0, C=1, S=2, E=3)
def gain_to_label(gain: float) -> int:
    if gain >= 1.0:
        return 3
    if gain >= 0.1:
        return 2
    if gain >= 0.01:
        return 1
    return 0


FEATURE_NAMES = [
    "bm25_score",
    "colbert_score",
    "title_query_overlap",
    "desc_query_overlap",
    "title_bigram_overlap",
    "brand_match",
    "title_length",
    "desc_length",
    "has_description",
    "query_length",
]

# Alias: config is the single source of truth for all hyperparameters
PARAMS = LAMBDAMART_PARAMS


def _bigrams(tokens: list) -> set:
    return set(zip(tokens, tokens[1:]))


def build_features(
    query: str,
    product_title: str,
    product_description: str,
    product_brand: str,
    bm25_score: float = 0.0,
    colbert_score: float = 0.0,
) -> list:
    q_tokens = query.lower().split()
    t_tokens = (product_title or "").lower().split()
    d_tokens = (product_description or "").lower().split()
    brand = (product_brand or "").lower()

    q_set = set(q_tokens)
    title_query_overlap = len(q_set & set(t_tokens)) / max(len(q_set), 1)
    desc_query_overlap = len(q_set & set(d_tokens)) / max(len(q_set), 1)

    q_bg = _bigrams(q_tokens)
    t_bg = _bigrams(t_tokens)
    title_bigram_overlap = len(q_bg & t_bg) / max(len(q_bg), 1) if q_bg else 0.0

    return [
        bm25_score,
        colbert_score,
        title_query_overlap,
        desc_query_overlap,
        title_bigram_overlap,
        float(bool(brand) and brand in query.lower()),
        np.log1p(len(t_tokens)),
        np.log1p(len(d_tokens)),
        float(bool(d_tokens)),
        float(len(q_tokens)),
    ]


class LambdaMARTRanker:
    def __init__(self):
        self.model: lgb.Booster | None = None

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        groups: list,
        eval_features: np.ndarray | None = None,
        eval_labels: np.ndarray | None = None,
        eval_groups: list | None = None,
        num_boost_round: int = LTR_NUM_BOOST_ROUND,
        early_stopping_rounds: int = LTR_EARLY_STOPPING_ROUNDS,
    ) -> None:
        train_set = lgb.Dataset(
            features, label=labels, group=groups, feature_name=FEATURE_NAMES
        )
        callbacks = [lgb.log_evaluation(period=50)]
        valid_sets = []

        if eval_features is not None:
            val_set = lgb.Dataset(
                eval_features,
                label=eval_labels,
                group=eval_groups,
                reference=train_set,
                feature_name=FEATURE_NAMES,
            )
            valid_sets.append(val_set)
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)
            )

        self.model = lgb.train(
            PARAMS,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets or None,
            callbacks=callbacks,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return self.model.predict(np.asarray(features, dtype=np.float32))

    def feature_importance(self) -> dict:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        scores = self.model.feature_importance("gain")
        return dict(sorted(zip(FEATURE_NAMES, scores), key=lambda x: -x[1]))

    def save(self, gcs_path: str, local_path: str = "/tmp/lambdamart.txt") -> None:
        from utils.gcs import upload
        self.model.save_model(local_path)
        upload(local_path, gcs_path)

    def load(self, gcs_path: str, local_path: str = "/tmp/lambdamart.txt") -> None:
        from utils.gcs import download
        download(gcs_path, local_path)
        self.model = lgb.Booster(model_file=local_path)
