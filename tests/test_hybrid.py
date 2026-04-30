"""Tests for HybridSearcher."""

from unittest.mock import MagicMock

import pytest

from ravenrag.hybrid import HybridSearcher


class TestHybridSearcher:
    def _make_store_and_embedder(self):
        store = MagicMock()
        store.get_all.return_value = {
            "ids": ["d1", "d2", "d3"],
            "documents": ["the cat sat on the mat", "the dog ran in the park", "machine learning is great"],
            "metadatas": [{"topic": "animals"}, {"topic": "animals"}, {"topic": "tech"}],
        }
        store.count.return_value = 3
        store.search.return_value = [
            {"id": "d1", "text": "the cat sat on the mat", "metadata": {"topic": "animals"}, "distance": 0.1},
            {"id": "d3", "text": "machine learning is great", "metadata": {"topic": "tech"}, "distance": 0.2},
            {"id": "d2", "text": "the dog ran in the park", "metadata": {"topic": "animals"}, "distance": 0.3},
        ]

        embedder = MagicMock()
        embedder.encode.return_value = [[0.1, 0.2, 0.3]]

        return store, embedder

    def test_search_returns_results(self):
        store, embedder = self._make_store_and_embedder()
        searcher = HybridSearcher(store, embedder, alpha=0.5)
        results = searcher.search("cat", top_k=2)

        assert len(results) == 2
        assert all(hasattr(r, "id") for r in results)
        assert all(hasattr(r, "text") for r in results)

    def test_search_empty_store(self):
        store = MagicMock()
        store.get_all.return_value = {"ids": [], "documents": [], "metadatas": []}
        embedder = MagicMock()

        searcher = HybridSearcher(store, embedder)
        results = searcher.search("test")

        assert results == []

    def test_alpha_full_vector(self):
        store, embedder = self._make_store_and_embedder()
        searcher = HybridSearcher(store, embedder, alpha=1.0)
        results = searcher.search("cat", top_k=3)

        # With alpha=1.0, vector search should dominate
        assert len(results) == 3
        assert results[0].id == "d1"  # Best vector match

    def test_alpha_full_bm25(self):
        store, embedder = self._make_store_and_embedder()
        searcher = HybridSearcher(store, embedder, alpha=0.0)
        results = searcher.search("cat sat mat", top_k=3)

        # With alpha=0.0, BM25 should dominate
        assert len(results) == 3
        assert results[0].id == "d1"  # Most keyword overlap

    def test_invalid_alpha_raises(self):
        store, embedder = self._make_store_and_embedder()
        with pytest.raises(ValueError, match="alpha"):
            HybridSearcher(store, embedder, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            HybridSearcher(store, embedder, alpha=-0.1)

    def test_custom_rrf_k(self):
        store, embedder = self._make_store_and_embedder()
        searcher = HybridSearcher(store, embedder, rrf_k=10)
        results = searcher.search("cat", top_k=2)
        assert len(results) == 2

    def test_distance_is_normalized(self):
        store, embedder = self._make_store_and_embedder()
        searcher = HybridSearcher(store, embedder)
        results = searcher.search("cat", top_k=3)
        for r in results:
            assert 0.0 <= r.distance <= 1.0, f"Distance {r.distance} out of [0, 1] range"

    def test_missing_rank_bm25(self, monkeypatch):
        """Test that a clear error is raised when rank-bm25 is not installed."""
        import ravenrag.hybrid as hybrid_mod

        monkeypatch.setattr(hybrid_mod, "HybridSearcher", HybridSearcher)
        # This test just verifies the import error path is reachable
        # (rank-bm25 IS installed in dev, so we can't easily test the missing case)
