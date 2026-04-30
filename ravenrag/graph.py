"""
KnowledgeGraph: Entity extraction and graph-based retrieval.

Extracts entities and relationships from documents, stores them in
an in-memory graph, and supports graph-traversal retrieval that
finds related documents through entity co-occurrence.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from .embed import EmbeddingBackend
    from .index import Document, QueryResult
    from .store import VectorStoreBackend

logger = logging.getLogger(__name__)

# Simple entity extraction patterns (noun-phrase heuristic)
_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"  # Capitalised phrases
)

# Common stopwords to skip as entities
_STOP_ENTITIES = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "Here",
        "There",
        "When",
        "Where",
        "Which",
        "What",
        "Who",
        "How",
        "However",
        "Also",
        "But",
        "And",
        "For",
        "From",
        "With",
        "About",
        "After",
        "Before",
        "Between",
        "Into",
        "Through",
        "During",
        "Without",
        "Again",
        "Further",
        "Then",
        "Once",
        "Each",
        "Every",
        "Both",
        "Few",
        "More",
        "Most",
        "Other",
        "Some",
        "Such",
        "Only",
        "Same",
        "Than",
        "Very",
        "Just",
        "Because",
        "Since",
        "While",
        "Although",
        "Though",
        "If",
        "Unless",
        "Until",
        "So",
        "Not",
        "No",
        "Nor",
        "As",
        "At",
        "By",
        "In",
        "Of",
        "On",
        "Or",
        "An",
        "Is",
        "It",
        "Its",
        "Do",
        "Did",
        "Has",
        "Had",
        "Have",
        "May",
        "Can",
        "Could",
        "Should",
        "Would",
        "Will",
        "Shall",
        "Must",
        "Need",
    }
)


@dataclass
class Entity:
    """A named entity extracted from text."""

    name: str
    entity_type: str = "CONCEPT"
    doc_ids: Set[str] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.name.lower() == other.name.lower()


@dataclass
class Relationship:
    """A co-occurrence relationship between two entities."""

    source: str
    target: str
    doc_ids: Set[str] = field(default_factory=set)
    weight: int = 0


class KnowledgeGraph:
    """In-memory knowledge graph built from document entities.

    Extracts capitalised noun phrases as entities, builds co-occurrence
    edges between entities that appear in the same document, and
    supports graph traversal to find related documents.

    Persistence is supported via JSON serialisation.

    Args:
        min_entity_length: Minimum character length for entities. Default 2.
    """

    def __init__(self, min_entity_length: int = 2):
        if min_entity_length < 1:
            raise ValueError("min_entity_length must be >= 1")
        self.min_entity_length = min_entity_length
        # name_lower → Entity
        self._entities: Dict[str, Entity] = {}
        # (source_lower, target_lower) → Relationship
        self._edges: Dict[Tuple[str, str], Relationship] = {}
        # doc_id → set of entity name_lower
        self._doc_entities: Dict[str, Set[str]] = defaultdict(set)

    def extract_entities(self, text: str) -> List[str]:
        """Extract entity names from text.

        Uses capitalised noun-phrase heuristic. Override this method
        to plug in spaCy, a transformer NER model, or an LLM extractor.

        Returns:
            List of entity name strings.
        """
        matches = _ENTITY_PATTERN.findall(text)
        entities = []
        for m in matches:
            m = m.strip()
            if len(m) < self.min_entity_length:
                continue
            if m in _STOP_ENTITIES:
                continue
            entities.append(m)
        return entities

    def add_document(self, doc: "Document") -> List[str]:
        """Extract entities from a document and add to the graph.

        Builds co-occurrence edges between all entity pairs found
        in the same document.

        Args:
            doc: A Document instance.

        Returns:
            List of entity names extracted.
        """
        entity_names = self.extract_entities(doc.text)
        if not entity_names:
            return []

        normalised: List[str] = []
        for name in entity_names:
            key = name.lower()
            normalised.append(key)
            if key not in self._entities:
                self._entities[key] = Entity(name=name)
            self._entities[key].doc_ids.add(doc.id)
            self._doc_entities[doc.id].add(key)

        # Build co-occurrence edges
        unique = list(set(normalised))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                a, b = sorted([unique[i], unique[j]])
                edge_key = (a, b)
                if edge_key not in self._edges:
                    self._edges[edge_key] = Relationship(
                        source=a,
                        target=b,
                    )
                self._edges[edge_key].doc_ids.add(doc.id)
                self._edges[edge_key].weight += 1

        return entity_names

    def add_documents(self, docs: List["Document"]) -> int:
        """Add multiple documents to the graph.

        Returns:
            Total number of entities extracted across all documents.
        """
        total = 0
        for doc in docs:
            total += len(self.add_document(doc))
        return total

    def get_entity(self, name: str) -> Optional[Entity]:
        """Look up an entity by name (case-insensitive)."""
        return self._entities.get(name.lower())

    def get_neighbours(self, entity_name: str, max_hops: int = 1) -> Dict[str, int]:
        """Get neighbouring entities within max_hops of the given entity.

        Args:
            entity_name: Entity to start from.
            max_hops: Maximum graph traversal depth. Default 1.

        Returns:
            Dict mapping entity name (lower) → hop distance.
        """
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")

        start = entity_name.lower()
        if start not in self._entities:
            return {}

        visited: Dict[str, int] = {start: 0}
        frontier = {start}

        for hop in range(1, max_hops + 1):
            next_frontier: Set[str] = set()
            for node in frontier:
                for (a, b), edge in self._edges.items():
                    neighbour = None
                    if a == node:
                        neighbour = b
                    elif b == node:
                        neighbour = a
                    if neighbour and neighbour not in visited:
                        visited[neighbour] = hop
                        next_frontier.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break

        # Remove the start node itself
        visited.pop(start, None)
        return visited

    def get_related_doc_ids(self, entity_name: str, max_hops: int = 1) -> List[str]:
        """Get document IDs related to an entity via graph traversal.

        Collects doc_ids from the entity itself and all neighbours
        within max_hops, ordered by hop distance (closest first).

        Args:
            entity_name: Entity to start from.
            max_hops: Maximum traversal depth.

        Returns:
            List of unique document IDs.
        """
        start = entity_name.lower()
        entity = self._entities.get(start)
        if not entity:
            return []

        # Collect doc_ids from entity itself first
        all_doc_ids: List[str] = list(entity.doc_ids)
        seen = set(all_doc_ids)

        # Then from neighbours, ordered by hop distance
        neighbours = self.get_neighbours(entity_name, max_hops=max_hops)
        for neighbour_name, _hop in sorted(neighbours.items(), key=lambda x: x[1]):
            neighbour = self._entities.get(neighbour_name)
            if neighbour:
                for did in neighbour.doc_ids:
                    if did not in seen:
                        all_doc_ids.append(did)
                        seen.add(did)

        return all_doc_ids

    @property
    def entity_count(self) -> int:
        """Number of unique entities in the graph."""
        return len(self._entities)

    @property
    def edge_count(self) -> int:
        """Number of edges (co-occurrence relationships)."""
        return len(self._edges)

    @property
    def entities(self) -> List[str]:
        """List of all entity names."""
        return [e.name for e in self._entities.values()]

    def save(self, path: str) -> None:
        """Persist the graph to a JSON file."""
        data = {
            "entities": {
                k: {"name": e.name, "type": e.entity_type, "doc_ids": sorted(e.doc_ids)}
                for k, e in self._entities.items()
            },
            "edges": {
                f"{a}||{b}": {"source": r.source, "target": r.target, "doc_ids": sorted(r.doc_ids), "weight": r.weight}
                for (a, b), r in self._edges.items()
            },
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load the graph from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._entities.clear()
        self._edges.clear()
        self._doc_entities.clear()

        for key, edata in data.get("entities", {}).items():
            entity = Entity(name=edata["name"], entity_type=edata.get("type", "CONCEPT"))
            entity.doc_ids = set(edata.get("doc_ids", []))
            self._entities[key] = entity
            for did in entity.doc_ids:
                self._doc_entities[did].add(key)

        for edge_key, rdata in data.get("edges", {}).items():
            parts = edge_key.split("||")
            if len(parts) == 2:
                a, b = parts
                rel = Relationship(
                    source=rdata["source"],
                    target=rdata["target"],
                    weight=rdata.get("weight", 0),
                )
                rel.doc_ids = set(rdata.get("doc_ids", []))
                self._edges[(a, b)] = rel

    def clear(self) -> None:
        """Remove all entities and relationships."""
        self._entities.clear()
        self._edges.clear()
        self._doc_entities.clear()


class GraphRetriever:
    """Retrieve documents using knowledge graph traversal.

    Combines entity extraction from the query with graph traversal
    to find related documents, then optionally merges with vector
    search results using reciprocal rank fusion.

    Args:
        graph: KnowledgeGraph instance.
        store: VectorStoreBackend (for fetching document content).
        embedder: EmbeddingBackend (for vector search fallback).
        max_hops: Maximum graph traversal depth. Default 2.
        alpha: Balance between graph (1.0) and vector (0.0). Default 0.5.
        rrf_k: RRF smoothing constant. Default 60.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        store: "VectorStoreBackend",
        embedder: "EmbeddingBackend",
        max_hops: int = 2,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        self.graph = graph
        self.store = store
        self.embedder = embedder
        self.max_hops = max_hops
        self.alpha = alpha
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List["QueryResult"]:
        """Search using graph traversal fused with vector similarity.

        1. Extract entities from the query.
        2. Traverse the graph to find related document IDs.
        3. Run vector search for semantic similarity.
        4. Fuse rankings with RRF.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            where: Optional metadata filter.

        Returns:
            List of QueryResult sorted by fused score.
        """
        from .index import QueryResult

        # 1. Graph-based retrieval
        query_entities = self.graph.extract_entities(query)
        graph_doc_ids: List[str] = []
        seen_ids: Set[str] = set()
        for entity_name in query_entities:
            for doc_id in self.graph.get_related_doc_ids(entity_name, max_hops=self.max_hops):
                if doc_id not in seen_ids:
                    graph_doc_ids.append(doc_id)
                    seen_ids.add(doc_id)

        # Also try lowercase query words as entity lookups
        for word in query.split():
            entity = self.graph.get_entity(word)
            if entity:
                for doc_id in self.graph.get_related_doc_ids(word, max_hops=self.max_hops):
                    if doc_id not in seen_ids:
                        graph_doc_ids.append(doc_id)
                        seen_ids.add(doc_id)

        # 2. Vector search
        query_embedding = self.embedder.encode([query])[0]
        vector_raw = self.store.search(query_embedding, top_k=top_k * 2, where=where)
        vector_results = [QueryResult(**r) for r in vector_raw]

        # 3. Build graph ranking (position in graph_doc_ids = rank)
        graph_rank: Dict[str, int] = {}
        for rank, doc_id in enumerate(graph_doc_ids):
            graph_rank[doc_id] = rank

        # 4. Build vector ranking
        vector_rank: Dict[str, int] = {}
        vector_by_id: Dict[str, "QueryResult"] = {}
        for rank, r in enumerate(vector_results):
            vector_rank[r.id] = rank
            vector_by_id[r.id] = r

        # 5. Fetch graph documents from store (if not already in vector results)
        graph_by_id: Dict[str, "QueryResult"] = {}
        if graph_doc_ids:
            fetch_ids = [did for did in graph_doc_ids if did not in vector_by_id]
            if fetch_ids:
                try:
                    fetched = self.store.get_by_ids(fetch_ids)
                    for i, fid in enumerate(fetched.get("ids") or []):
                        docs_list = fetched.get("documents") or []
                        metas_list = fetched.get("metadatas") or []
                        if i < len(docs_list) and docs_list[i]:
                            meta = metas_list[i] if i < len(metas_list) and metas_list[i] else {}
                            # Apply where-filter
                            if where and not all(meta.get(k) == v for k, v in where.items()):
                                continue
                            graph_by_id[fid] = QueryResult(
                                id=fid,
                                text=docs_list[i],
                                metadata=meta,
                                distance=1.0,  # No distance from graph
                            )
                except Exception:
                    logger.warning("Failed to fetch graph documents", exc_info=True)

        # 6. RRF fusion
        all_ids = set(graph_rank.keys()) | set(vector_rank.keys())
        scored: List[Tuple[str, float]] = []

        for doc_id in all_ids:
            graph_score = 0.0
            vector_score = 0.0
            if doc_id in graph_rank:
                graph_score = 1.0 / (self.rrf_k + graph_rank[doc_id])
            if doc_id in vector_rank:
                vector_score = 1.0 / (self.rrf_k + vector_rank[doc_id])
            fused = self.alpha * graph_score + (1.0 - self.alpha) * vector_score
            scored.append((doc_id, fused))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 7. Build final results
        results: List[QueryResult] = []
        for doc_id, fused_score in scored[:top_k]:
            if doc_id in vector_by_id:
                r = vector_by_id[doc_id]
                r.distance = 1.0 - fused_score  # Convert to distance
                results.append(r)
            elif doc_id in graph_by_id:
                r = graph_by_id[doc_id]
                r.distance = 1.0 - fused_score
                results.append(r)

        return results
