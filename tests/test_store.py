"""Tests for VectorStore and VectorStoreBackend protocol."""

from ravenrag.store import VectorStore, VectorStoreBackend


class TestVectorStoreBackend:
    def test_vectorstore_implements_protocol(self, tmp_path):
        store = VectorStore(persist_dir=str(tmp_path / "db"))
        assert isinstance(store, VectorStoreBackend)

    def test_protocol_has_required_methods(self):
        """Verify VectorStoreBackend defines the expected interface."""
        methods = {"upsert", "search", "delete", "count", "get_all", "clear"}
        for method in methods:
            assert hasattr(VectorStoreBackend, method)


class TestVectorStore:
    def test_count_empty(self, tmp_path):
        store = VectorStore(persist_dir=str(tmp_path / "db"))
        assert store.count() == 0

    def test_get_all_empty(self, tmp_path):
        store = VectorStore(persist_dir=str(tmp_path / "db"))
        result = store.get_all()
        assert result["ids"] == []

    def test_search_empty(self, tmp_path):
        store = VectorStore(persist_dir=str(tmp_path / "db"))
        results = store.search([0.1, 0.2, 0.3], top_k=5)
        assert results == []

    def test_clear(self, tmp_path):
        store = VectorStore(persist_dir=str(tmp_path / "db"))
        store.clear()
        assert store.count() == 0
