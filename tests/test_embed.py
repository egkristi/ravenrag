"""Tests for Embedder."""

from ravenrag.embed import Embedder


class TestEmbedder:
    def test_encode(self):
        embedder = Embedder()
        embeddings = embedder.encode(["Hello world", "Another sentence"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0
        assert isinstance(embeddings[0], list)
        assert isinstance(embeddings[0][0], float)
