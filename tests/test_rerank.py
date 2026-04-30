"""Tests for Reranker."""

from unittest.mock import MagicMock

from ravenrag.index import QueryResult
from ravenrag.rerank import Reranker


class TestReranker:
    def test_rerank_sorts_by_score(self):
        reranker = Reranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        reranker._model = mock_model

        results = [
            QueryResult(id="1", text="low", metadata={}, distance=0.1),
            QueryResult(id="2", text="high", metadata={}, distance=0.2),
            QueryResult(id="3", text="mid", metadata={}, distance=0.3),
        ]
        reranked = reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].id == "2"
        assert reranked[0].rerank_score == 0.9
        assert reranked[1].id == "3"

    def test_rerank_empty(self):
        reranker = Reranker()
        assert reranker.rerank("query", [], top_k=5) == []

    def test_rerank_sets_scores(self):
        reranker = Reranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.7, 0.3]
        reranker._model = mock_model

        results = [
            QueryResult(id="a", text="aa", metadata={}, distance=0.1),
            QueryResult(id="b", text="bb", metadata={}, distance=0.2),
        ]
        reranked = reranker.rerank("q", results)

        assert all(r.rerank_score is not None for r in reranked)

    def test_lazy_model_loading(self):
        reranker = Reranker()
        assert reranker._model is None
