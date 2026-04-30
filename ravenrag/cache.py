"""
Cache: LRU caching for embeddings and query results.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, List, Optional, Tuple


class EmbeddingCache:
    """Thread-safe LRU cache for embedding computations.

    Caches embeddings keyed by (text, model_name) so identical queries
    return instantly without re-computing.

    Args:
        maxsize: Maximum number of cached embeddings. 0 disables caching.
    """

    def __init__(self, maxsize: int = 1024):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding, or None if not cached."""
        if self.maxsize <= 0:
            return None
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        if self.maxsize <= 0:
            return
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
            self._cache[key] = embedding

    def get_or_compute(
        self,
        texts: List[str],
        compute_fn: Any,
    ) -> List[List[float]]:
        """Get embeddings from cache or compute missing ones.

        Args:
            texts: Texts to embed.
            compute_fn: Callable that takes List[str] and returns List[List[float]].

        Returns:
            List of embeddings in the same order as texts.
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        to_compute: List[Tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results[i] = cached
            else:
                to_compute.append((i, text))

        if to_compute:
            missing_texts = [t for _, t in to_compute]
            computed = compute_fn(missing_texts)
            for (idx, text), emb in zip(to_compute, computed):
                self.put(text, emb)
                results[idx] = emb

        return results  # type: ignore[return-value]

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)
