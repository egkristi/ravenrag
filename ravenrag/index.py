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

# Lazy import to avoid circular deps
_EmbeddingCache = None


def _get_cache_class():
    global _EmbeddingCache
    if _EmbeddingCache is None:
        from .cache import EmbeddingCache

        _EmbeddingCache = EmbeddingCache
    return _EmbeddingCache


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
        store: Optional[Any] = None,
        cache_size: int = 1024,
    ):
        if embedding_backend is not None:
            self.embedder: EmbeddingBackend = embedding_backend
        else:
            self.embedder = Embedder(model_name=embedding_model)
        if store is not None:
            self.store = store
        else:
            self.store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
        self.batch_size = batch_size
        self._reranker = None
        CacheClass = _get_cache_class()
        self._embedding_cache = CacheClass(maxsize=cache_size)

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

        query_embedding = self._embedding_cache.get_or_compute([query], self.embedder.encode)[0]
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

        # Try to fetch full parent text from the store (via protocol method)
        parent_ids = list(seen.keys())
        parent_results: List[QueryResult] = []
        try:
            parent_data = self.store.get_by_ids(parent_ids)
        except Exception:
            logger.warning("Failed to fetch parent documents, falling back to chunks", exc_info=True)
            parent_data = {"ids": [], "documents": [], "metadatas": []}

        fetched = {}
        for i, pid in enumerate(parent_data.get("ids") or []):
            docs = parent_data.get("documents") or []
            metas = parent_data.get("metadatas") or []
            if i < len(docs) and docs[i]:
                fetched[pid] = (docs[i], metas[i] if i < len(metas) else {})

        for parent_id, best_chunk in seen.items():
            if parent_id in fetched:
                text, metadata = fetched[parent_id]
                parent_results.append(
                    QueryResult(
                        id=parent_id,
                        text=text,
                        metadata=metadata or {},
                        distance=best_chunk.distance,
                    )
                )
            else:
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

    # ------------------------------------------------------------------
    # Async API — wraps sync methods via asyncio.to_thread
    # ------------------------------------------------------------------

    async def aadd(self, documents: List[Document]) -> None:
        """Async version of :meth:`add`."""
        import asyncio

        await asyncio.to_thread(self.add, documents)

    async def aquery(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        rerank: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> List[QueryResult]:
        """Async version of :meth:`query`."""
        import asyncio

        return await asyncio.to_thread(self.query, query, top_k, where, rerank, rerank_model)

    async def ahybrid_query(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        alpha: float = 0.5,
    ) -> List[QueryResult]:
        """Async version of :meth:`hybrid_query`."""
        import asyncio

        return await asyncio.to_thread(self.hybrid_query, query, top_k, where, alpha)

    def graph_query(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        graph: Optional[Any] = None,
        max_hops: int = 2,
        alpha: float = 0.5,
    ) -> List[QueryResult]:
        """Knowledge-graph-augmented retrieval.

        Extracts entities from the query, traverses the knowledge graph
        to find related documents, and fuses the graph signal with
        vector similarity via reciprocal rank fusion.

        Args:
            query: Search query.
            top_k: Number of results to return.
            where: Optional metadata filter.
            graph: A KnowledgeGraph instance. Required.
            max_hops: Graph traversal depth. Default 2.
            alpha: Balance between graph (1.0) and vector (0.0). Default 0.5.

        Returns:
            List of QueryResult ranked by fused score.
        """
        if graph is None:
            raise ValueError("A KnowledgeGraph instance is required for graph_query")
        from .graph import GraphRetriever

        retriever = GraphRetriever(
            graph=graph,
            store=self.store,
            embedder=self.embedder,
            max_hops=max_hops,
            alpha=alpha,
        )
        return retriever.search(query, top_k=top_k, where=where)

    def query_stream(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        rerank: bool = False,
    ):
        """Generator that yields results one at a time.

        Useful for streaming results to clients or processing
        results incrementally.
        """
        results = self.query(query, top_k=top_k, where=where, rerank=rerank)
        yield from results


class MultiCollectionRouter:
    """Search across multiple DocumentIndex instances.

    Routes queries to all collections and merges results by distance.

    Args:
        indices: Dict mapping collection name to DocumentIndex.
    """

    def __init__(self, indices: Dict[str, "DocumentIndex"]):
        if not indices:
            raise ValueError("At least one index is required")
        self.indices = indices

    def query(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        collections: Optional[List[str]] = None,
    ) -> List[QueryResult]:
        """Query across multiple collections, merge by distance.

        Args:
            query: Search query.
            top_k: Total results to return.
            where: Metadata filter.
            collections: Subset of collections to search. None = all.
        """
        all_results: List[QueryResult] = []
        target = collections or list(self.indices.keys())

        for name in target:
            if name not in self.indices:
                logger.warning("Collection '%s' not found, skipping", name)
                continue
            idx = self.indices[name]
            results = idx.query(query, top_k=top_k, where=where)
            for r in results:
                r.metadata["_collection"] = name
            all_results.extend(results)

        all_results.sort(key=lambda r: r.distance)
        return all_results[:top_k]
