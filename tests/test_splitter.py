"""Tests for TextSplitter and TokenSplitter."""

from unittest.mock import MagicMock

import pytest

from ravenrag import Document
from ravenrag.splitter import TextSplitter, TokenSplitter


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
