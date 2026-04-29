"""Tests for RavenRAG DocumentIndex."""

import pytest
import tempfile
import shutil
from ravenrag import DocumentIndex, Document


class TestDocumentIndex:
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
