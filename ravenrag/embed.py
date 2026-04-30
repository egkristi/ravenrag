"""
Embedder: Embedding backends for local and remote models.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import List, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol that all embedding backends must satisfy."""

    def encode(self, texts: List[str]) -> List[List[float]]: ...

    def encode_batched(self, texts: List[str], batch_size: int = 64) -> List[List[float]]: ...


class Embedder:
    """Local embedding backend using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def encode_batched(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Encode texts in batches to manage memory usage."""
        model = self._get_model()
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings


class OllamaBackend:
    """Embedding backend using Ollama's local API.

    Requires a running Ollama instance with an embedding model pulled.
    Example: ``ollama pull nomic-embed-text``
    """

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        data = json.dumps({"model": self.model_name, "input": texts}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                result = json.loads(resp.read())
                return result["embeddings"]
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}: {e}") from e

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts via Ollama API."""
        return self._embed(texts)

    def encode_batched(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Encode texts in batches via Ollama API."""
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed(batch))
        return all_embeddings
