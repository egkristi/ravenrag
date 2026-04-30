"""
Reranker: Cross-encoder reranking for improved search quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .index import QueryResult


class Reranker:
    """Rerank search results using a cross-encoder model.

    Cross-encoders evaluate query-document pairs jointly, producing
    more accurate relevance scores than bi-encoder similarity alone.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for reranking. Install with: pip install sentence-transformers"
                ) from None
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load reranker model '{self.model_name}': {e}. "
                    "Check that the model name is a valid cross-encoder."
                ) from e
        return self._model

    def rerank(self, query: str, results: List[QueryResult], top_k: int = 5) -> List[QueryResult]:
        """Rerank results by cross-encoder relevance score.

        Args:
            query: The search query.
            results: Initial search results to rerank.
            top_k: Number of top results to return.

        Returns:
            Reranked list of QueryResult with rerank_score set.
        """
        if not results:
            return []

        model = self._get_model()
        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs)

        for r, score in zip(results, scores):
            r.rerank_score = float(score)

        reranked = sorted(results, key=lambda r: r.rerank_score or 0.0, reverse=True)
        return reranked[:top_k]
