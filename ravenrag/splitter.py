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
