"""
Query router — pre-retrieval, rule-based, zero LLM calls.

Determines retrieval strategy based on query token count:
  ≤2 tokens  → hybrid (BM25 + FAISS): short queries need semantic expansion
  3-4 tokens → hybrid (BM25 + FAISS): standard path
  ≥5 tokens  → BM25 only: long queries are specific enough for lexical matching;
                FAISS adds semantic noise by retrieving attribute-mismatched products

The routing decision is ~0ms overhead (integer comparison).
"""


class QueryRouter:
    def classify(self, query: str) -> str:
        n = len(query.split())
        if n <= 2:
            return "short"
        if n <= 4:
            return "standard"
        return "long"

    def use_faiss(self, query: str) -> bool:
        return self.classify(query) != "long"
