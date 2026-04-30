"""Tests for RavenRAG DocumentIndex."""

import shutil
import tempfile
from unittest.mock import patch

import pytest

from ravenrag import Document, DocumentIndex


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
