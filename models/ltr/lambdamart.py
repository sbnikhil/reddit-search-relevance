import numpy as np
import lightgbm as lgb
from google.cloud import storage

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

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 5, 10],
    "label_gain": [0, 1, 3, 7],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}


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
        float(bool(brand) and brand in query.lower()),   # brand_match
        np.log1p(len(t_tokens)),                         # title_length
        np.log1p(len(d_tokens)),                         # desc_length
        float(bool(d_tokens)),                           # has_description
        float(len(q_tokens)),                            # query_length
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
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
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
        self.model.save_model(local_path)
        bucket = gcs_path.replace("gs://", "").split("/")[0]
        blob = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
        storage.Client().bucket(bucket).blob(blob).upload_from_filename(local_path)
        print(f"Model saved -> {gcs_path}")

    def load(self, gcs_path: str, local_path: str = "/tmp/lambdamart.txt") -> None:
        bucket = gcs_path.replace("gs://", "").split("/")[0]
        blob = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
        storage.Client().bucket(bucket).blob(blob).download_to_filename(local_path)
        self.model = lgb.Booster(model_file=local_path)
        print(f"Model loaded <- {gcs_path}")
