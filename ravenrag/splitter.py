"""
TextSplitter: Split long documents into overlapping chunks.
"""

from typing import List

from .index import Document


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
