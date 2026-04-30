"""
DocumentIndex: Core indexing and retrieval logic.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .embed import Embedder
from .store import VectorStore

if TYPE_CHECKING:
    from .embed import EmbeddingBackend

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """A single search result with typed fields."""

    id: str
    text: str
    metadata: Dict
    distance: float
    rerank_score: Optional[float] = None

    @property
    def citation(self) -> str:
        """Return a human-readable source citation.

        Uses metadata fields (source, filename, chunk_index) to build
        a citation string like ``docs/auth.md#chunk3``.
        """
        source = self.metadata.get("source", self.metadata.get("filename", self.id[:12]))
        chunk = self.metadata.get("chunk_index")
        if chunk is not None:
            return f"{source}#chunk{chunk}"
        return str(source)

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for backwards compatibility."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like .get() for backwards compatibility."""
        return getattr(self, key, default)


class Document:
    """A simple document container."""

    def __init__(self, text: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None):
        self.text = text
        self.metadata = metadata or {}
        self.id = doc_id or hashlib.sha256(text.encode()).hexdigest()

    def __repr__(self):
        return f"Document(id={self.id[:8]}..., text={self.text[:50]}...)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class DocumentIndex:
    """Main interface for indexing documents and querying them."""

    def __init__(
        self,
        persist_dir: str = "./ravenrag_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_backend: Optional["EmbeddingBackend"] = None,
        collection_name: str = "documents",
        batch_size: int = 64,
    ):
        if embedding_backend is not None:
            self.embedder: EmbeddingBackend = embedding_backend
        else:
            self.embedder = Embedder(model_name=embedding_model)
        self.store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
        self.batch_size = batch_size
        self._reranker = None

    def add(self, documents: List[Document]) -> None:
        """Index a list of documents (batched for large sets)."""
        if not documents:
            return
        for doc in documents:
            if not doc.text or not doc.text.strip():
                raise ValueError(f"Document {doc.id} has empty text")
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            texts = [doc.text for doc in batch]
            embeddings = self.embedder.encode_batched(texts, batch_size=self.batch_size)
            self.store.upsert(batch, embeddings)

    def query(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        rerank: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> List[QueryResult]:
        """Query the index. Optionally rerank results with a cross-encoder."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        query_embedding = self.embedder.encode([query])[0]
        fetch_k = top_k * 4 if rerank else top_k
        raw = self.store.search(query_embedding, top_k=fetch_k, where=where)
        results = [QueryResult(**r) for r in raw]

        if rerank and results:
            if self._reranker is None or self._reranker.model_name != rerank_model:
                from .rerank import Reranker

                self._reranker = Reranker(model_name=rerank_model)
            results = self._reranker.rerank(query, results, top_k=top_k)

        return results[:top_k]

    def hybrid_query(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        alpha: float = 0.5,
    ) -> List[QueryResult]:
        """Hybrid search combining vector similarity with BM25 keyword matching.

        Requires: ``pip install 'ravenrag[hybrid]'``

        Args:
            alpha: Balance between vector (1.0) and BM25 (0.0). Default 0.5.
        """
        from .hybrid import HybridSearcher

        searcher = HybridSearcher(self.store, self.embedder, alpha=alpha)
        return searcher.search(query, top_k=top_k, where=where)

    def query_for_prompt(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        template: Optional[str] = None,
    ) -> str:
        """Query and return a formatted context string ready for LLM prompts."""
        from .context import ContextFormatter

        results = self.query(query, top_k=top_k, where=where)
        formatter = ContextFormatter(template=template)
        return formatter.format(query, results)

    def query_parent(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        rerank: bool = False,
    ) -> List[QueryResult]:
        """Search on chunk level but return parent documents.

        When documents have been split into chunks (via TextSplitter or
        SemanticSplitter), this retrieves the matching chunks, then
        fetches the full parent document for each match.

        Chunks must have ``source_id`` in their metadata (set automatically
        by all built-in splitters).

        Returns one result per unique parent document, with the full
        parent text and the best chunk's distance score.
        """
        results = self.query(query, top_k=top_k * 3, where=where, rerank=rerank)

        # Deduplicate by source_id, keep the best (lowest distance) per parent
        seen: Dict[str, QueryResult] = {}
        for r in results:
            parent_id = r.metadata.get("source_id", r.id)
            if parent_id not in seen or r.distance < seen[parent_id].distance:
                seen[parent_id] = r

        # Try to fetch full parent text from the store
        parent_results: List[QueryResult] = []
        for parent_id, best_chunk in seen.items():
            try:
                parent_data = self.store.collection.get(ids=[parent_id], include=["documents", "metadatas"])
                if parent_data["ids"] and parent_data["documents"]:
                    parent_results.append(
                        QueryResult(
                            id=parent_id,
                            text=parent_data["documents"][0],
                            metadata=parent_data["metadatas"][0] if parent_data.get("metadatas") else {},
                            distance=best_chunk.distance,
                        )
                    )
                else:
                    # Parent not in store (was chunked before indexing), return chunk
                    parent_results.append(best_chunk)
            except Exception:
                logger.warning("Failed to fetch parent %s, falling back to chunk", parent_id, exc_info=True)
                parent_results.append(best_chunk)

        parent_results.sort(key=lambda r: r.distance)
        return parent_results[:top_k]

    def delete(self, doc_id: str) -> None:
        """Remove a document by ID."""
        self.store.delete(doc_id)

    def count(self) -> int:
        """Return the number of indexed documents."""
        return self.store.count()

    def clear(self) -> None:
        """Delete all documents."""
        self.store.clear()
