"""Tests for TextSplitter."""

import pytest

from ravenrag import Document
from ravenrag.splitter import TextSplitter


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
