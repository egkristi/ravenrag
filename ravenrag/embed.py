"""
Embedder: Sentence-transformers wrapper for local embeddings.
"""

from typing import List
import numpy as np

# Lazy import to avoid heavy loading at import time
_embedder = None
_model_name = None


def _get_embedder(model_name: str):
    global _embedder, _model_name
    if _embedder is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_name)
        _model_name = model_name
    return _embedder


class Embedder:
    """Local embedding model wrapper."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        model = _get_embedder(self.model_name)
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
