"""
Pipeline: Composable retrieval pipeline for RavenRAG.

Build complex indexing and retrieval flows declaratively::

    from ravenrag import Pipeline
    from ravenrag.loaders import load_directory
    from ravenrag.splitter import SemanticSplitter
    from ravenrag.index import DocumentIndex

    pipe = Pipeline(
        loader=load_directory,
        splitter=SemanticSplitter(embedder=embedder),
        index=DocumentIndex(persist_dir="./db"),
    )
    pipe.run("./documents/")
    results = pipe.query("What is RAG?")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional

if TYPE_CHECKING:
    from .index import Document, DocumentIndex, QueryResult

logger = logging.getLogger(__name__)


class Pipeline:
    """Composable pipeline for indexing and querying documents.

    Args:
        loader: Callable that takes a path and returns List[Document].
        splitter: Object with ``split_documents(docs) -> docs`` method. Optional.
        index: A DocumentIndex instance for storage and retrieval.
        on_error: Error handling strategy: ``"skip"`` (default), ``"raise"``, or
            a callable ``(error, doc) -> None``.
    """

    def __init__(
        self,
        loader: Callable[..., List["Document"]],
        index: "DocumentIndex",
        splitter: Optional[Any] = None,
        on_error: str | Callable = "skip",
    ):
        self.loader = loader
        self.splitter = splitter
        self.index = index
        self.on_error = on_error
        self._stats: Dict[str, int] = {"loaded": 0, "chunked": 0, "indexed": 0, "errors": 0}

    @property
    def stats(self) -> Dict[str, int]:
        """Return pipeline run statistics."""
        return dict(self._stats)

    def run(self, path: str, **loader_kwargs: Any) -> Dict[str, int]:
        """Run the full indexing pipeline.

        Args:
            path: Path to load documents from.
            **loader_kwargs: Additional arguments passed to the loader.

        Returns:
            Dict with keys: loaded, chunked, indexed, errors.
        """
        self._stats = {"loaded": 0, "chunked": 0, "indexed": 0, "errors": 0}

        # Load
        try:
            docs = self.loader(path, **loader_kwargs)
        except Exception as e:
            self._handle_error(e, None)
            return self.stats
        self._stats["loaded"] = len(docs)

        # Split
        if self.splitter is not None:
            try:
                docs = self.splitter.split_documents(docs)
            except Exception as e:
                self._handle_error(e, None)
                return self.stats
        self._stats["chunked"] = len(docs)

        # Index
        try:
            self.index.add(docs)
            self._stats["indexed"] = len(docs)
        except Exception as e:
            self._handle_error(e, None)

        logger.info(
            "Pipeline complete: loaded=%d, chunked=%d, indexed=%d, errors=%d",
            self._stats["loaded"],
            self._stats["chunked"],
            self._stats["indexed"],
            self._stats["errors"],
        )
        return self.stats

    def query(self, query: str, top_k: int = 5, **kwargs: Any) -> List["QueryResult"]:
        """Query the pipeline's index."""
        return self.index.query(query, top_k=top_k, **kwargs)

    def query_stream(self, query: str, top_k: int = 5, **kwargs: Any) -> Generator["QueryResult", None, None]:
        """Stream results one at a time (generator-based)."""
        results = self.index.query(query, top_k=top_k, **kwargs)
        yield from results

    def _handle_error(self, error: Exception, doc: Any) -> None:
        self._stats["errors"] += 1
        if self.on_error == "raise":
            raise error
        elif callable(self.on_error):
            self.on_error(error, doc)
        else:
            logger.warning("Pipeline error (skipping): %s", error, exc_info=True)
