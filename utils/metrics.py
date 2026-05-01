import math

# Single source of truth for ESCI relevance levels.
# Matches LambdaMART's label_gain=[0,1,3,7] config.
GAIN_MAP = {'E': 3, 'S': 1, 'C': 0, 'I': 0, None: 0}
_LEVEL_TO_DCG_GAIN = {0: 0, 1: 1, 3: 7}  # 2^level - 1


def ndcg_at_k(ranked_labels, k=10):
    """
    ranked_labels: ESCI label strings ('E','S','C','I') or None, in predicted rank order.
    Returns NDCG@k using gain=[0,1,3,7] — same objective as LambdaMART training.
    """
    def _dcg(labels):
        s = 0.0
        for i, lbl in enumerate(labels[:k]):
            level = GAIN_MAP.get(lbl, 0)
            s += _LEVEL_TO_DCG_GAIN[level] / math.log2(i + 2)
        return s

    ideal = sorted(ranked_labels, key=lambda l: GAIN_MAP.get(l, 0), reverse=True)
    idcg = _dcg(ideal)
    return _dcg(ranked_labels) / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_labels, k=10):
    """Reciprocal rank of first Exact match."""
    for i, lbl in enumerate(ranked_labels[:k]):
        if lbl == 'E':
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked_labels, k=10):
    """Fraction of top-k results that are Exact or Substitute."""
    return sum(1 for lbl in ranked_labels[:k] if GAIN_MAP.get(lbl, 0) > 0) / k


def gain_to_label(gain: float) -> str:
    """Convert ESCI float gain (1.0/0.1/0.01/0.0) to label string."""
    if gain >= 1.0:
        return 'E'
    if gain >= 0.1:
        return 'S'
    if gain >= 0.01:
        return 'C'
    return 'I'
