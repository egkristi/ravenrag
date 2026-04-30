"""Tests for RavenRAG DocumentIndex."""

import shutil
import tempfile
from unittest.mock import patch

import pytest

from ravenrag import Document, DocumentIndex
from ravenrag.index import QueryResult


class TestQueryResult:
    def test_dict_access(self):
        r = QueryResult(id="1", text="hello", metadata={"k": "v"}, distance=0.5)
        assert r["id"] == "1"
        assert r["text"] == "hello"
        assert r["metadata"] == {"k": "v"}
        assert r["distance"] == 0.5

    def test_dot_access(self):
        r = QueryResult(id="1", text="hello", metadata={}, distance=0.5)
        assert r.id == "1"
        assert r.text == "hello"

    def test_get_with_default(self):
        r = QueryResult(id="1", text="hello", metadata={}, distance=0.5)
        assert r.get("rerank_score") is None
        assert r.get("nonexistent", "default") == "default"

    def test_dict_key_error(self):
        r = QueryResult(id="1", text="hello", metadata={}, distance=0.5)
        with pytest.raises(KeyError):
            r["nonexistent_key"]

    def test_citation_with_source(self):
        r = QueryResult(id="1", text="text", metadata={"source": "docs/auth.md"}, distance=0.1)
        assert r.citation == "docs/auth.md"

    def test_citation_with_chunk(self):
        r = QueryResult(
            id="1",
            text="text",
            metadata={"source": "docs/auth.md", "chunk_index": 3},
            distance=0.1,
        )
        assert r.citation == "docs/auth.md#chunk3"

    def test_citation_fallback_to_id(self):
        r = QueryResult(id="abc123def456", text="text", metadata={}, distance=0.1)
        assert r.citation == "abc123def456"


class TestDocument:
    def test_id_from_sha256(self):
        doc = Document("hello")
        assert len(doc.id) == 64  # sha256 hex digest

    def test_custom_id(self):
        doc = Document("hello", doc_id="custom-1")
        assert doc.id == "custom-1"

    def test_same_text_same_id(self):
        d1 = Document("same text")
        d2 = Document("same text")
        assert d1.id == d2.id

    def test_metadata_defaults_to_empty(self):
        doc = Document("test")
        assert doc.metadata == {}


class TestDocumentIndex:
    """Unit tests with mocked embedder."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("ravenrag.index.Embedder")
    def test_add_calls_embedder(self, MockEmbedder):
        mock_embedder = MockEmbedder.return_value
        mock_embedder.encode_batched.return_value = [[0.1, 0.2, 0.3]]

        index = DocumentIndex(persist_dir=self.temp_dir)
        index.embedder = mock_embedder
        index.add([Document("Test doc")])

        mock_embedder.encode_batched.assert_called_once()

    @patch("ravenrag.index.Embedder")
    def test_query_calls_embedder(self, MockEmbedder):
        mock_embedder = MockEmbedder.return_value
        mock_embedder.encode_batched.return_value = [[0.1, 0.2, 0.3]]
        mock_embedder.encode.return_value = [[0.1, 0.2, 0.3]]

        index = DocumentIndex(persist_dir=self.temp_dir)
        index.embedder = mock_embedder
        index.add([Document("The sky is blue.")])
        index.query("sky")

        mock_embedder.encode_batched.assert_called_once()
        mock_embedder.encode.assert_called_once_with(["sky"])

    def test_add_empty_list(self):
        index = DocumentIndex(persist_dir=self.temp_dir)
        index.add([])  # Should not raise
        assert index.count() == 0

    def test_add_empty_text_raises(self):
        index = DocumentIndex(persist_dir=self.temp_dir)
        with pytest.raises(ValueError, match="empty text"):
            index.add([Document("  ", doc_id="blank")])

    def test_collection_name(self):
        idx1 = DocumentIndex(persist_dir=self.temp_dir, collection_name="col1")
        idx2 = DocumentIndex(persist_dir=self.temp_dir, collection_name="col2")
        assert idx1.store.collection.name == "col1"
        assert idx2.store.collection.name == "col2"

    def test_query_empty_string_raises(self):
        index = DocumentIndex(persist_dir=self.temp_dir)
        with pytest.raises(ValueError, match="non-empty"):
            index.query("")

    def test_query_whitespace_only_raises(self):
        index = DocumentIndex(persist_dir=self.temp_dir)
        with pytest.raises(ValueError, match="non-empty"):
            index.query("   ")

    def test_query_negative_top_k_raises(self):
        index = DocumentIndex(persist_dir=self.temp_dir)
        with pytest.raises(ValueError, match="top_k"):
            index.query("test", top_k=-1)


@pytest.mark.integration
class TestDocumentIndexIntegration:
    """Integration tests that load real models."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.index = DocumentIndex(persist_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_query(self):
        docs = [
            Document("The sky is blue."),
            Document("Grass is green."),
            Document("The sun is hot."),
        ]
        self.index.add(docs)
        assert self.index.count() == 3

        results = self.index.query("What color is the sky?", top_k=1)
        assert len(results) == 1
        assert "sky" in results[0]["text"].lower()

    def test_query_with_metadata_filter(self):
        docs = [
            Document("Python is great", metadata={"lang": "python"}),
            Document("JavaScript is fast", metadata={"lang": "js"}),
        ]
        self.index.add(docs)

        results = self.index.query("programming", top_k=2, where={"lang": "python"})
        assert all(r["metadata"]["lang"] == "python" for r in results)

    def test_delete(self):
        doc = Document("Temporary document.", doc_id="temp-1")
        self.index.add([doc])
        assert self.index.count() == 1

        self.index.delete("temp-1")
        assert self.index.count() == 0

    def test_clear(self):
        docs = [Document(f"Doc {i}") for i in range(5)]
        self.index.add(docs)
        assert self.index.count() == 5

        self.index.clear()
        assert self.index.count() == 0
