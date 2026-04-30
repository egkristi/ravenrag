"""
HybridSearcher: Combine vector search with BM25 keyword matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from .index import QueryResult

if TYPE_CHECKING:
    from .embed import EmbeddingBackend
    from .store import VectorStore


class HybridSearcher:
    """Combine vector similarity search with BM25 keyword matching.

    Uses Reciprocal Rank Fusion (RRF) to merge ranking signals from
    both vector similarity and BM25 keyword scores.

    Requires: ``pip install 'ravenrag[hybrid]'``
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingBackend,
        alpha: float = 0.5,
    ):
        self.store = store
        self.embedder = embedder
        self.alpha = alpha

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[QueryResult]:
        """Search using both vector similarity and BM25 keyword matching.

        Args:
            query: Search query.
            top_k: Number of results to return.
            where: Optional metadata filter (applied to vector search only).

        Returns:
            List of QueryResult ranked by fused score.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for hybrid search. Install with: pip install 'ravenrag[hybrid]'"
            ) from None

        # Retrieve all documents for BM25
        all_docs = self.store.get_all()
        if not all_docs["ids"]:
            return []

        doc_ids = all_docs["ids"]
        doc_texts = all_docs["documents"]
        doc_metadatas = all_docs.get("metadatas") or [{}] * len(doc_ids)

        # BM25 scoring
        tokenized = [text.lower().split() for text in doc_texts]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())

        # Vector search (fetch more than needed for fusion)
        query_embedding = self.embedder.encode([query])[0]
        total = self.store.count()
        vector_results = self.store.search(
            query_embedding,
            top_k=min(total, top_k * 4),
            where=where,
        )

        # Reciprocal Rank Fusion (k=60 is standard)
        rrf_k = 60

        # Vector RRF scores
        vector_rrf: Dict[str, float] = {}
        for rank, r in enumerate(vector_results):
            vector_rrf[r["id"]] = 1.0 / (rrf_k + rank + 1)

        # BM25 RRF scores
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_rrf: Dict[str, float] = {}
        for rank, idx in enumerate(bm25_ranked):
            bm25_rrf[doc_ids[idx]] = 1.0 / (rrf_k + rank + 1)

        # Fuse scores
        all_candidate_ids = set(vector_rrf.keys()) | set(bm25_rrf.keys())
        fused = []
        for doc_id in all_candidate_ids:
            v = vector_rrf.get(doc_id, 0.0)
            b = bm25_rrf.get(doc_id, 0.0)
            score = self.alpha * v + (1.0 - self.alpha) * b
            fused.append((doc_id, score))

        fused.sort(key=lambda x: x[1], reverse=True)

        # Build result objects
        doc_map = {doc_ids[i]: (doc_texts[i], doc_metadatas[i] or {}) for i in range(len(doc_ids))}
        results = []
        for doc_id, score in fused[:top_k]:
            text, metadata = doc_map[doc_id]
            results.append(
                QueryResult(
                    id=doc_id,
                    text=text,
                    metadata=metadata,
                    distance=1.0 / score if score > 0 else float("inf"),
                )
            )

        return results
