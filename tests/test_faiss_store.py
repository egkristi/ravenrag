"""Tests for FAISS vector store backend."""

import pytest

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")

from ravenrag.stores.faiss_store import FaissStore  # noqa: E402


class _FakeDoc:
    def __init__(self, id, text, metadata=None):
        self.id = id
        self.text = text
        self.metadata = metadata or {}


DIM = 4


def _emb(vals):
    """Create an embedding padded/truncated to DIM."""
    return (vals + [0.0] * DIM)[:DIM]


class TestFaissStore:
    def test_count_empty(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        assert store.count() == 0

    def test_upsert_and_count(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        docs = [_FakeDoc("a", "hello"), _FakeDoc("b", "world")]
        embs = [_emb([1.0, 0.0]), _emb([0.0, 1.0])]
        store.upsert(docs, embs)
        assert store.count() == 2

    def test_upsert_empty(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        store.upsert([], [])
        assert store.count() == 0

    def test_search(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        docs = [
            _FakeDoc("a", "hello"),
            _FakeDoc("b", "world"),
            _FakeDoc("c", "test"),
        ]
        embs = [_emb([1.0, 0.0]), _emb([0.0, 1.0]), _emb([0.5, 0.5])]
        store.upsert(docs, embs)

        results = store.search(_emb([1.0, 0.0]), top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "a"

    def test_search_empty(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        results = store.search(_emb([1.0, 0.0]))
        assert results == []

    def test_search_with_where(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        docs = [
            _FakeDoc("a", "hello", {"topic": "greet"}),
            _FakeDoc("b", "world", {"topic": "place"}),
        ]
        embs = [_emb([1.0, 0.0]), _emb([0.0, 1.0])]
        store.upsert(docs, embs)

        results = store.search(_emb([0.5, 0.5]), top_k=5, where={"topic": "place"})
        assert len(results) == 1
        assert results[0]["id"] == "b"

    def test_delete(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        docs = [_FakeDoc("a", "hello"), _FakeDoc("b", "world")]
        embs = [_emb([1.0, 0.0]), _emb([0.0, 1.0])]
        store.upsert(docs, embs)

        store.delete("a")
        assert store.count() == 1
        assert store.get_all()["ids"] == ["b"]

    def test_delete_nonexistent(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        store.delete("nonexistent")  # Should not raise

    def test_get_all(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        docs = [_FakeDoc("a", "hello"), _FakeDoc("b", "world")]
        embs = [_emb([1.0, 0.0]), _emb([0.0, 1.0])]
        store.upsert(docs, embs)

        data = store.get_all()
        assert set(data["ids"]) == {"a", "b"}
        assert len(data["documents"]) == 2

    def test_get_by_ids(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        docs = [_FakeDoc("a", "hello"), _FakeDoc("b", "world")]
        embs = [_emb([1.0, 0.0]), _emb([0.0, 1.0])]
        store.upsert(docs, embs)

        result = store.get_by_ids(["b"])
        assert result["ids"] == ["b"]
        assert result["documents"] == ["world"]

    def test_get_by_ids_empty(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        result = store.get_by_ids(["missing"])
        assert result["ids"] == []

    def test_clear(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        docs = [_FakeDoc("a", "hello")]
        embs = [_emb([1.0, 0.0])]
        store.upsert(docs, embs)

        store.clear()
        assert store.count() == 0

    def test_upsert_overwrites(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        store.upsert([_FakeDoc("a", "v1")], [_emb([1.0, 0.0])])
        store.upsert([_FakeDoc("a", "v2")], [_emb([0.0, 1.0])])
        assert store.count() == 1
        assert store.get_all()["documents"] == ["v2"]

    def test_persistence(self, tmp_path):
        store1 = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        store1.upsert([_FakeDoc("a", "hello")], [_emb([1.0, 0.0])])

        # Load from disk
        store2 = FaissStore(persist_dir=str(tmp_path), dimension=DIM)
        assert store2.count() == 1
        assert store2.get_all()["ids"] == ["a"]

    def test_auto_dimension(self, tmp_path):
        store = FaissStore(persist_dir=str(tmp_path))  # No dimension given
        store.upsert([_FakeDoc("a", "hello")], [_emb([1.0, 0.0])])
        assert store.count() == 1
        assert store._dimension == DIM
