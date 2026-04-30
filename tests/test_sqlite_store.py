"""Tests for the SqliteVecStore backend."""

import shutil
import tempfile

from ravenrag.index import Document
from ravenrag.stores.sqlite_store import SqliteVecStore


class TestSqliteVecStore:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = SqliteVecStore(persist_dir=self.temp_dir, collection_name="test")

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_count_empty(self):
        assert self.store.count() == 0

    def test_upsert_and_count(self):
        docs = [Document(text="hello", metadata={"source": "a"})]
        embeddings = [[1.0, 2.0, 3.0]]
        self.store.upsert(docs, embeddings)
        assert self.store.count() == 1

    def test_search(self):
        docs = [
            Document(text="hello world", metadata={"source": "a"}),
            Document(text="foo bar", metadata={"source": "b"}),
        ]
        emb = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        self.store.upsert(docs, emb)

        results = self.store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "hello world"

    def test_search_with_where(self):
        docs = [
            Document(text="hello", metadata={"source": "a"}),
            Document(text="world", metadata={"source": "b"}),
        ]
        emb = [[1.0, 0.0], [0.0, 1.0]]
        self.store.upsert(docs, emb)

        results = self.store.search([1.0, 0.0], top_k=5, where={"source": "b"})
        assert len(results) == 1
        assert results[0]["metadata"]["source"] == "b"

    def test_delete(self):
        docs = [Document(text="hello", metadata={})]
        self.store.upsert(docs, [[1.0, 2.0]])
        assert self.store.count() == 1
        self.store.delete(docs[0].id)
        assert self.store.count() == 0

    def test_get_all(self):
        docs = [Document(text="a"), Document(text="b")]
        self.store.upsert(docs, [[1.0], [2.0]])
        result = self.store.get_all()
        assert len(result["ids"]) == 2

    def test_get_by_ids(self):
        docs = [Document(text="a", doc_id="id1"), Document(text="b", doc_id="id2")]
        self.store.upsert(docs, [[1.0], [2.0]])
        result = self.store.get_by_ids(["id1"])
        assert len(result["ids"]) == 1
        assert result["documents"][0] == "a"

    def test_get_by_ids_empty(self):
        result = self.store.get_by_ids([])
        assert result["ids"] == []

    def test_clear(self):
        docs = [Document(text="hello")]
        self.store.upsert(docs, [[1.0]])
        self.store.clear()
        assert self.store.count() == 0

    def test_upsert_overwrites(self):
        doc = Document(text="v1", doc_id="same")
        self.store.upsert([doc], [[1.0]])
        doc2 = Document(text="v2", doc_id="same")
        self.store.upsert([doc2], [[2.0]])
        assert self.store.count() == 1
        result = self.store.get_by_ids(["same"])
        assert result["documents"][0] == "v2"

    def test_search_empty(self):
        results = self.store.search([1.0, 2.0], top_k=5)
        assert results == []
