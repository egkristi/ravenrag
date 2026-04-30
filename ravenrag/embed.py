"""
Embedder: Sentence-transformers wrapper for local embeddings.
"""

from typing import List


class Embedder:
    """Local embedding model wrapper with instance-level caching."""

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
