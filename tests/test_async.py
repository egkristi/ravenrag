"""Tests for async API methods."""

import asyncio
from unittest.mock import MagicMock

from ravenrag.cache import EmbeddingCache
from ravenrag.index import Document, DocumentIndex


class TestAsyncAPI:
    def _make_index(self):
        idx = DocumentIndex.__new__(DocumentIndex)
        idx.store = MagicMock()
        idx.embedder = MagicMock()
        idx.batch_size = 64
        idx._reranker = None
        idx._embedding_cache = EmbeddingCache(maxsize=10)
        return idx

    def test_aquery(self):
        idx = self._make_index()
        idx.embedder.encode.return_value = [[1.0, 2.0]]
        idx.store.search.return_value = [{"id": "1", "text": "result", "metadata": {}, "distance": 0.1}]

        results = asyncio.run(idx.aquery("test"))
        assert len(results) == 1
        assert results[0].id == "1"

    def test_aadd(self):
        idx = self._make_index()
        idx.embedder.encode_batched.return_value = [[1.0, 2.0]]

        docs = [Document(text="hello", metadata={})]
        asyncio.run(idx.aadd(docs))

        idx.store.upsert.assert_called_once()

    def test_ahybrid_query(self):
        idx = self._make_index()
        idx.store.get_all.return_value = {
            "ids": ["1"],
            "documents": ["doc text"],
            "metadatas": [{}],
        }
        idx.store.count.return_value = 1
        idx.embedder.encode.return_value = [[1.0, 2.0]]
        idx.store.search.return_value = [{"id": "1", "text": "doc text", "metadata": {}, "distance": 0.1}]

        results = asyncio.run(idx.ahybrid_query("test"))
        assert isinstance(results, list)
