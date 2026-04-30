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
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                ) from None
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {e}. "
                    "Check that the model name is valid and you have internet access for first download."
                ) from e
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
                body = resp.read()
        except urllib.error.HTTPError as e:
            raise ConnectionError(f"Ollama returned HTTP {e.code} for model '{self.model_name}': {e.reason}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}: {e}") from e

        try:
            result = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ollama returned invalid JSON: {e}") from e

        if "embeddings" not in result:
            raise ValueError(f"Ollama response missing 'embeddings' key. Got keys: {list(result.keys())}")

        return result["embeddings"]

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


class OpenAIBackend:
    """Embedding backend for any OpenAI-compatible ``/v1/embeddings`` endpoint.

    Works with vLLM, LM Studio, LocalAI, text-generation-inference,
    Xinference, and any other server that speaks the OpenAI embeddings API.

    Example usage::

        # Local vLLM server
        backend = OpenAIBackend(
            model_name="BAAI/bge-base-en-v1.5",
            base_url="http://localhost:8000/v1",
        )

        # LM Studio
        backend = OpenAIBackend(
            model_name="nomic-embed-text-v1.5",
            base_url="http://localhost:1234/v1",
        )

        # OpenAI cloud (needs API key)
        backend = OpenAIBackend(
            model_name="text-embedding-3-small",
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
        )
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _embed(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model_name, "input": texts}).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                body = resp.read()
        except urllib.error.HTTPError as e:
            raise ConnectionError(
                f"OpenAI-compatible API returned HTTP {e.code} for model '{self.model_name}': {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to {self.base_url}: {e}") from e

        try:
            result = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"API returned invalid JSON: {e}") from e

        if "data" not in result:
            raise ValueError(f"Response missing 'data' key. Got keys: {list(result.keys())}")

        # Sort by index to guarantee order matches input
        sorted_data = sorted(result["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in sorted_data]

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts via OpenAI-compatible embeddings API."""
        return self._embed(texts)

    def encode_batched(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Encode texts in batches via OpenAI-compatible embeddings API."""
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed(batch))
        return all_embeddings


class VLLMBackend(OpenAIBackend):
    """Embedding backend for a local vLLM server.

    Thin wrapper around :class:`OpenAIBackend` with vLLM-friendly defaults.

    Start vLLM with::

        vllm serve BAAI/bge-base-en-v1.5 --task embed

    Then::

        backend = VLLMBackend("BAAI/bge-base-en-v1.5")
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
    ):
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key)
