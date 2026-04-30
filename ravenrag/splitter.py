"""
TextSplitter: Split long documents into overlapping chunks.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

from .index import Document

if TYPE_CHECKING:
    from .embed import EmbeddingBackend


class TextSplitter:
    """Split text into chunks with configurable size and overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, separator: str = "\n"):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split a single text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Try to break at separator boundary
            if end < len(text):
                break_point = text.rfind(self.separator, start, end)
                if break_point > start:
                    end = break_point + len(self.separator)

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        # Remove empty chunks
        return [c for c in chunks if c]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunked documents, preserving metadata."""
        result: List[Document] = []
        for doc in documents:
            chunks = self.split_text(doc.text)
            for i, chunk in enumerate(chunks):
                metadata = {**doc.metadata, "chunk_index": i, "source_id": doc.id}
                result.append(Document(text=chunk, metadata=metadata))
        return result


class TokenSplitter:
    """Split text into chunks based on token count rather than character count.

    Uses the tokenizer from the embedding model for accurate token counting.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError(
                    "transformers is required for TokenSplitter. Install with: pip install transformers"
                ) from None
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except OSError:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{self.model_name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load tokenizer for '{self.model_name}': {e}") from e
        return self._tokenizer

    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks based on token count."""
        tokenizer = self._get_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end == len(tokens):
                break
            start = end - self.chunk_overlap

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunked documents based on token count."""
        result: List[Document] = []
        for doc in documents:
            chunks = self.split_text(doc.text)
            for i, chunk in enumerate(chunks):
                metadata = {**doc.metadata, "chunk_index": i, "source_id": doc.id}
                result.append(Document(text=chunk, metadata=metadata))
        return result


class SemanticSplitter:
    """Split text at semantic boundaries using embedding similarity.

    Instead of splitting at fixed character/token positions, this splitter:
    1. Splits text into sentences.
    2. Embeds each sentence.
    3. Measures cosine similarity between consecutive sentences.
    4. Cuts where similarity drops below a threshold.
    5. Merges adjacent sentences into chunks (respecting max_chunk_size).

    Args:
        embedder: An EmbeddingBackend to compute sentence embeddings.
        threshold: Cosine similarity threshold (0.0–1.0). Lower = larger chunks.
        max_chunk_size: Maximum characters per chunk (hard limit).
        min_chunk_size: Minimum characters per chunk (merge small chunks).
    """

    def __init__(
        self,
        embedder: EmbeddingBackend,
        threshold: float = 0.5,
        max_chunk_size: int = 2048,
        min_chunk_size: int = 100,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Embed all sentences
        embeddings = self.embedder.encode(sentences)

        # Find split points where similarity drops below threshold
        split_indices: List[int] = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.threshold:
                split_indices.append(i + 1)

        # Build chunks from split points
        chunks: List[str] = []
        start = 0
        for idx in split_indices:
            chunk = " ".join(sentences[start:idx])
            if chunk:
                chunks.append(chunk)
            start = idx
        # Last chunk
        chunk = " ".join(sentences[start:])
        if chunk:
            chunks.append(chunk)

        # Enforce max_chunk_size: split oversized chunks
        final: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                final.append(chunk)
            else:
                # Fall back to character-based splitting for oversized chunks
                for i in range(0, len(chunk), self.max_chunk_size):
                    part = chunk[i : i + self.max_chunk_size].strip()
                    if part:
                        final.append(part)

        # Merge tiny chunks with their neighbors
        merged: List[str] = []
        for chunk in final:
            if merged and len(merged[-1]) < self.min_chunk_size:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        return merged if merged else [text]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents at semantic boundaries, preserving metadata."""
        result: List[Document] = []
        for doc in documents:
            chunks = self.split_text(doc.text)
            for i, chunk in enumerate(chunks):
                metadata = {**doc.metadata, "chunk_index": i, "source_id": doc.id, "split_method": "semantic"}
                result.append(Document(text=chunk, metadata=metadata))
        return result
