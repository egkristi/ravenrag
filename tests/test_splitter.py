"""Tests for TextSplitter, TokenSplitter, and SemanticSplitter."""

from unittest.mock import MagicMock

import pytest

from ravenrag import Document
from ravenrag.splitter import SemanticSplitter, TextSplitter, TokenSplitter


class TestTextSplitter:
    def test_short_text_no_split(self):
        splitter = TextSplitter(chunk_size=100)
        chunks = splitter.split_text("Short text")
        assert chunks == ["Short text"]

    def test_splits_long_text(self):
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5, separator=" ")
        text = "one two three four five six seven eight"
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        # All original content should be recoverable
        for word in text.split():
            assert any(word in c for c in chunks)

    def test_overlap(self):
        splitter = TextSplitter(chunk_size=10, chunk_overlap=3, separator="")
        text = "abcdefghijklmnop"
        chunks = splitter.split_text(text)
        # Each chunk except the first should overlap with the previous
        for i in range(1, len(chunks)):
            assert chunks[i][:3] == chunks[i - 1][-3:] or len(chunks[i - 1]) < 10

    def test_split_documents(self):
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
        docs = [Document("A" * 50, metadata={"key": "val"})]
        result = splitter.split_documents(docs)
        assert len(result) > 1
        assert all(r.metadata["key"] == "val" for r in result)
        assert all("chunk_index" in r.metadata for r in result)
        assert all("source_id" in r.metadata for r in result)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=10, chunk_overlap=10)


class TestTokenSplitter:
    def _mock_tokenizer(self):
        """Create a mock tokenizer that splits on spaces."""
        tok = MagicMock()
        tok.encode.side_effect = lambda text, **kw: list(range(len(text.split())))
        tok.decode.side_effect = lambda tokens, **kw: " ".join(f"w{t}" for t in tokens)
        return tok

    def test_short_text_no_split(self):
        splitter = TokenSplitter(chunk_size=100, chunk_overlap=10)
        splitter._tokenizer = self._mock_tokenizer()
        chunks = splitter.split_text("hello world")
        assert chunks == ["hello world"]

    def test_splits_long_text(self):
        splitter = TokenSplitter(chunk_size=3, chunk_overlap=1)
        splitter._tokenizer = self._mock_tokenizer()
        text = "a b c d e f g h"
        chunks = splitter.split_text(text)
        assert len(chunks) > 1

    def test_split_documents(self):
        splitter = TokenSplitter(chunk_size=3, chunk_overlap=1)
        splitter._tokenizer = self._mock_tokenizer()
        docs = [Document("a b c d e f", metadata={"key": "val"})]
        result = splitter.split_documents(docs)
        assert len(result) > 1
        assert all(r.metadata["key"] == "val" for r in result)
        assert all("chunk_index" in r.metadata for r in result)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            TokenSplitter(chunk_size=10, chunk_overlap=10)


class TestSemanticSplitter:
    def _mock_embedder(self):
        """Embedder that returns different vectors for different sentences."""
        embedder = MagicMock()
        # Return vectors where adjacent sentences are similar, but a shift happens at sentence 3
        embedder.encode.side_effect = lambda texts: [[1.0, 0.0] if i < 2 else [0.0, 1.0] for i in range(len(texts))]
        return embedder

    def test_short_text_no_split(self):
        embedder = self._mock_embedder()
        splitter = SemanticSplitter(embedder, threshold=0.5)
        chunks = splitter.split_text("Short text.")
        assert chunks == ["Short text."]

    def test_splits_at_semantic_boundary(self):
        embedder = MagicMock()
        # Sentences 0,1 similar; sentence 2 very different
        embedder.encode.return_value = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        splitter = SemanticSplitter(embedder, threshold=0.8, min_chunk_size=1)
        text = "First sentence. Second sentence. Totally different topic."
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_respects_max_chunk_size(self):
        embedder = MagicMock()
        # All similar → no semantic splits, but max_chunk_size forces splits
        embedder.encode.return_value = [[1.0, 0.0]] * 5
        splitter = SemanticSplitter(embedder, threshold=0.1, max_chunk_size=50, min_chunk_size=1)
        text = "A" * 30 + ". " + "B" * 30 + ". " + "C" * 30 + ". " + "D" * 30 + ". " + "E" * 30 + "."
        chunks = splitter.split_text(text)
        # All chunks should respect max_chunk_size
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_merges_tiny_chunks(self):
        embedder = MagicMock()
        # Every pair is dissimilar → many splits
        embedder.encode.return_value = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        splitter = SemanticSplitter(embedder, threshold=0.9, min_chunk_size=200)
        text = "A. B. C."
        chunks = splitter.split_text(text)
        # Should merge small chunks
        assert len(chunks) <= 2

    def test_split_documents_preserves_metadata(self):
        embedder = MagicMock()
        embedder.encode.return_value = [[1.0, 0.0], [0.0, 1.0]]
        splitter = SemanticSplitter(embedder, threshold=0.5)
        docs = [Document("First topic. Different topic.", metadata={"key": "val"})]
        result = splitter.split_documents(docs)
        assert len(result) >= 1
        assert all(r.metadata["key"] == "val" for r in result)
        assert all("chunk_index" in r.metadata for r in result)
        assert all(r.metadata.get("split_method") == "semantic" for r in result)

    def test_empty_text(self):
        embedder = MagicMock()
        splitter = SemanticSplitter(embedder)
        chunks = splitter.split_text("")
        assert chunks == [] or chunks == [""]

    def test_cosine_similarity_static(self):
        assert SemanticSplitter._cosine_similarity([1, 0], [1, 0]) == 1.0
        assert SemanticSplitter._cosine_similarity([1, 0], [0, 1]) == 0.0
        assert abs(SemanticSplitter._cosine_similarity([1, 1], [1, 1]) - 1.0) < 1e-9

    def test_cosine_similarity_zero_vector(self):
        assert SemanticSplitter._cosine_similarity([0, 0], [1, 0]) == 0.0
