"""Tests for the MultiCollectionRouter and query_stream."""

from unittest.mock import MagicMock

from ravenrag.index import MultiCollectionRouter, QueryResult


class TestMultiCollectionRouter:
    def _mock_index(self, results):
        idx = MagicMock()
        idx.query.return_value = results
        return idx

    def test_query_merges_results(self):
        idx1 = self._mock_index(
            [
                QueryResult(id="a", text="from idx1", metadata={}, distance=0.1),
            ]
        )
        idx2 = self._mock_index(
            [
                QueryResult(id="b", text="from idx2", metadata={}, distance=0.05),
            ]
        )

        router = MultiCollectionRouter({"docs": idx1, "code": idx2})
        results = router.query("test", top_k=5)

        assert len(results) == 2
        # Best result first (lowest distance)
        assert results[0].id == "b"
        assert results[0].metadata["_collection"] == "code"

    def test_query_respects_top_k(self):
        idx = self._mock_index(
            [QueryResult(id=f"d{i}", text=f"t{i}", metadata={}, distance=float(i)) for i in range(10)]
        )
        router = MultiCollectionRouter({"all": idx})
        results = router.query("test", top_k=3)
        assert len(results) == 3

    def test_query_specific_collections(self):
        idx1 = self._mock_index([QueryResult(id="a", text="a", metadata={}, distance=0.1)])
        idx2 = self._mock_index([QueryResult(id="b", text="b", metadata={}, distance=0.2)])

        router = MultiCollectionRouter({"docs": idx1, "code": idx2})
        results = router.query("test", collections=["docs"])

        idx1.query.assert_called_once()
        idx2.query.assert_not_called()
        assert len(results) == 1

    def test_empty_indices_raises(self):
        try:
            MultiCollectionRouter({})
            assert False, "Should have raised"
        except ValueError:
            pass


class TestQueryStream:
    def test_stream_yields_all_results(self):
        from ravenrag.index import DocumentIndex

        idx = MagicMock(spec=DocumentIndex)
        idx.query.return_value = [
            QueryResult(id="1", text="a", metadata={}, distance=0.1),
            QueryResult(id="2", text="b", metadata={}, distance=0.2),
        ]

        # Call query_stream directly on the mock index
        # We need to test the actual method, so let's use a real instance with a mock store
        from ravenrag.index import DocumentIndex as RealIndex

        real_idx = RealIndex.__new__(RealIndex)
        real_idx.store = MagicMock()
        real_idx.embedder = MagicMock()
        real_idx.embedder.encode.return_value = [[1.0, 2.0]]
        real_idx.batch_size = 64
        real_idx._reranker = None

        from ravenrag.cache import EmbeddingCache

        real_idx._embedding_cache = EmbeddingCache(maxsize=10)

        real_idx.store.search.return_value = [
            {"id": "1", "text": "a", "metadata": {}, "distance": 0.1},
            {"id": "2", "text": "b", "metadata": {}, "distance": 0.2},
        ]

        streamed = list(real_idx.query_stream("test", top_k=2))
        assert len(streamed) == 2
        assert streamed[0].id == "1"
