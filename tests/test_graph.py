"""Tests for KnowledgeGraph and GraphRetriever."""

from unittest.mock import MagicMock

import pytest

from ravenrag.graph import GraphRetriever, KnowledgeGraph
from ravenrag.index import Document, QueryResult


class TestKnowledgeGraph:
    def test_extract_entities_capitalised(self):
        g = KnowledgeGraph()
        entities = g.extract_entities("Python is a language created by Guido van Rossum.")
        assert "Python" in entities
        assert "Guido" in entities

    def test_extract_entities_skips_stopwords(self):
        g = KnowledgeGraph()
        entities = g.extract_entities("The quick brown fox. This is a test.")
        assert "The" not in entities
        assert "This" not in entities

    def test_extract_entities_multi_word(self):
        g = KnowledgeGraph()
        entities = g.extract_entities("Machine Learning is used in Natural Language Processing.")
        assert "Machine Learning" in entities
        assert "Natural Language Processing" in entities

    def test_extract_entities_min_length(self):
        g = KnowledgeGraph(min_entity_length=5)
        entities = g.extract_entities("Go and Rust are languages.")
        # "Go" is length 2, should be filtered; "Rust" is length 4, also filtered
        assert "Go" not in entities
        assert "Rust" not in entities

    def test_min_entity_length_validation(self):
        with pytest.raises(ValueError, match="min_entity_length"):
            KnowledgeGraph(min_entity_length=0)

    def test_add_document_creates_entities(self):
        g = KnowledgeGraph()
        doc = Document("Python was created by Guido van Rossum.", doc_id="d1")
        names = g.add_document(doc)
        assert len(names) > 0
        assert g.entity_count > 0
        entity = g.get_entity("Python")
        assert entity is not None
        assert "d1" in entity.doc_ids

    def test_add_document_creates_edges(self):
        g = KnowledgeGraph()
        doc = Document("Python was created by Guido van Rossum.", doc_id="d1")
        g.add_document(doc)
        assert g.edge_count > 0

    def test_add_document_empty_text(self):
        g = KnowledgeGraph()
        doc = Document("no capitalised words here at all", doc_id="d1")
        names = g.add_document(doc)
        assert names == []
        assert g.entity_count == 0

    def test_add_documents_multiple(self):
        g = KnowledgeGraph()
        docs = [
            Document("Python is great.", doc_id="d1"),
            Document("Rust is fast.", doc_id="d2"),
        ]
        total = g.add_documents(docs)
        assert total >= 2
        assert g.entity_count >= 2

    def test_get_entity_case_insensitive(self):
        g = KnowledgeGraph()
        g.add_document(Document("Python rocks.", doc_id="d1"))
        assert g.get_entity("python") is not None
        assert g.get_entity("PYTHON") is not None
        assert g.get_entity("Python") is not None

    def test_get_entity_not_found(self):
        g = KnowledgeGraph()
        assert g.get_entity("Nonexistent") is None

    def test_get_neighbours(self):
        g = KnowledgeGraph()
        g.add_document(Document("Alice met Bob at the Conference.", doc_id="d1"))
        g.add_document(Document("Bob works with Carol.", doc_id="d2"))

        neighbours = g.get_neighbours("Alice", max_hops=1)
        assert "bob" in neighbours
        assert "conference" in neighbours

    def test_get_neighbours_multi_hop(self):
        g = KnowledgeGraph()
        g.add_document(Document("Alice met Bob.", doc_id="d1"))
        g.add_document(Document("Bob met Carol.", doc_id="d2"))

        # Alice -> Bob (1 hop), Bob -> Carol (2 hops)
        neighbours = g.get_neighbours("Alice", max_hops=2)
        assert "bob" in neighbours
        assert "carol" in neighbours
        assert neighbours["bob"] == 1
        assert neighbours["carol"] == 2

    def test_get_neighbours_invalid_hops(self):
        g = KnowledgeGraph()
        with pytest.raises(ValueError, match="max_hops"):
            g.get_neighbours("Alice", max_hops=0)

    def test_get_neighbours_unknown_entity(self):
        g = KnowledgeGraph()
        assert g.get_neighbours("Unknown") == {}

    def test_get_related_doc_ids(self):
        g = KnowledgeGraph()
        g.add_document(Document("Alice met Bob.", doc_id="d1"))
        g.add_document(Document("Bob met Carol.", doc_id="d2"))

        doc_ids = g.get_related_doc_ids("Alice", max_hops=2)
        assert "d1" in doc_ids
        assert "d2" in doc_ids

    def test_get_related_doc_ids_unknown(self):
        g = KnowledgeGraph()
        assert g.get_related_doc_ids("Unknown") == []

    def test_entities_property(self):
        g = KnowledgeGraph()
        g.add_document(Document("Python and Rust are great.", doc_id="d1"))
        names = g.entities
        assert isinstance(names, list)
        assert len(names) >= 2

    def test_save_and_load(self, tmp_path):
        g = KnowledgeGraph()
        g.add_document(Document("Alice met Bob.", doc_id="d1"))
        g.add_document(Document("Bob met Carol.", doc_id="d2"))

        path = str(tmp_path / "graph.json")
        g.save(path)

        g2 = KnowledgeGraph()
        g2.load(path)

        assert g2.entity_count == g.entity_count
        assert g2.edge_count == g.edge_count
        assert g2.get_entity("alice") is not None
        assert "d1" in g2.get_entity("alice").doc_ids

    def test_clear(self):
        g = KnowledgeGraph()
        g.add_document(Document("Alice met Bob.", doc_id="d1"))
        assert g.entity_count > 0
        g.clear()
        assert g.entity_count == 0
        assert g.edge_count == 0


class TestGraphRetriever:
    def _make_store_and_embedder(self):
        store = MagicMock()
        store.search.return_value = [
            {"id": "d1", "text": "Python is great", "metadata": {}, "distance": 0.1},
            {"id": "d3", "text": "Machine Learning rocks", "metadata": {}, "distance": 0.2},
        ]
        store.get_by_ids.return_value = {
            "ids": ["d2"],
            "documents": ["Bob met Carol"],
            "metadatas": [{}],
        }

        embedder = MagicMock()
        embedder.encode.return_value = [[0.1, 0.2, 0.3]]

        return store, embedder

    def test_search_returns_results(self):
        store, embedder = self._make_store_and_embedder()
        graph = KnowledgeGraph()
        graph.add_document(Document("Python is great", doc_id="d1"))
        graph.add_document(Document("Bob uses Python", doc_id="d2"))

        retriever = GraphRetriever(graph, store, embedder, max_hops=2)
        results = retriever.search("Python", top_k=5)

        assert len(results) > 0
        assert all(isinstance(r, QueryResult) for r in results)

    def test_search_empty_graph(self):
        store, embedder = self._make_store_and_embedder()
        graph = KnowledgeGraph()

        retriever = GraphRetriever(graph, store, embedder)
        results = retriever.search("test", top_k=5)

        # Should still return vector results
        assert len(results) > 0

    def test_search_with_where_filter(self):
        store = MagicMock()
        store.search.return_value = [
            {"id": "d1", "text": "Python is great", "metadata": {"topic": "code"}, "distance": 0.1},
        ]
        store.get_by_ids.return_value = {
            "ids": ["d2"],
            "documents": ["Bob uses Python"],
            "metadatas": [{"topic": "people"}],
        }
        embedder = MagicMock()
        embedder.encode.return_value = [[0.1, 0.2]]

        graph = KnowledgeGraph()
        graph.add_document(Document("Python is great", doc_id="d1"))
        graph.add_document(Document("Bob uses Python", doc_id="d2"))

        retriever = GraphRetriever(graph, store, embedder)
        results = retriever.search("Python", top_k=5, where={"topic": "code"})

        # d2 has topic=people, should be filtered from graph results
        ids = [r.id for r in results]
        assert "d2" not in ids

    def test_invalid_alpha(self):
        store, embedder = self._make_store_and_embedder()
        graph = KnowledgeGraph()

        with pytest.raises(ValueError, match="alpha"):
            GraphRetriever(graph, store, embedder, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            GraphRetriever(graph, store, embedder, alpha=-0.1)

    def test_alpha_full_graph(self):
        store, embedder = self._make_store_and_embedder()
        graph = KnowledgeGraph()
        graph.add_document(Document("Python is great", doc_id="d1"))

        retriever = GraphRetriever(graph, store, embedder, alpha=1.0)
        results = retriever.search("Python", top_k=5)
        assert len(results) > 0

    def test_alpha_full_vector(self):
        store, embedder = self._make_store_and_embedder()
        graph = KnowledgeGraph()

        retriever = GraphRetriever(graph, store, embedder, alpha=0.0)
        results = retriever.search("Python", top_k=5)
        # Pure vector — should return vector results
        assert len(results) > 0


class TestDocumentIndexGraphQuery:
    def test_graph_query_requires_graph(self):
        from ravenrag.index import DocumentIndex

        idx = MagicMock(spec=DocumentIndex)
        idx.graph_query = DocumentIndex.graph_query.__get__(idx, DocumentIndex)

        with pytest.raises(ValueError, match="KnowledgeGraph"):
            idx.graph_query("test")

    def test_graph_query_delegates_to_retriever(self):
        from ravenrag.index import DocumentIndex

        store = MagicMock()
        store.search.return_value = [
            {"id": "d1", "text": "Python is great", "metadata": {}, "distance": 0.1},
        ]
        store.get_by_ids.return_value = {"ids": [], "documents": [], "metadatas": []}

        embedder = MagicMock()
        embedder.encode.return_value = [[0.1, 0.2, 0.3]]
        embedder.encode_batched.return_value = [[0.1, 0.2, 0.3]]

        idx = MagicMock(spec=DocumentIndex)
        idx.store = store
        idx.embedder = embedder
        idx.graph_query = DocumentIndex.graph_query.__get__(idx, DocumentIndex)

        graph = KnowledgeGraph()
        graph.add_document(Document("Python is great", doc_id="d1"))

        results = idx.graph_query("Python", graph=graph, top_k=5)
        assert len(results) > 0
