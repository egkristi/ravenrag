"""
ContextFormatter: Format search results for LLM prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .index import QueryResult


class ContextFormatter:
    """Format search results into context strings for LLM prompts.

    Example::

        formatter = ContextFormatter()
        prompt = formatter.format("What is RAG?", results)
        # Send prompt to your LLM
    """

    DEFAULT_TEMPLATE = "Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n"

    DEFAULT_CHUNK_TEMPLATE = "[{i}] {text}\n"

    def __init__(
        self,
        template: Optional[str] = None,
        chunk_template: Optional[str] = None,
        include_sources: bool = True,
    ):
        self.template = template or self.DEFAULT_TEMPLATE
        self.chunk_template = chunk_template or self.DEFAULT_CHUNK_TEMPLATE
        self.include_sources = include_sources

    def format(self, query: str, results: List[QueryResult]) -> str:
        """Format query and results into a prompt-ready string."""
        parts = []
        for i, r in enumerate(results, 1):
            chunk = self.chunk_template.format(i=i, text=r.text)
            if self.include_sources:
                source = r.metadata.get("source", r.metadata.get("filename", ""))
                if source:
                    chunk += f"(source: {source})\n"
            parts.append(chunk)

        context = "\n".join(parts)
        return self.template.format(context=context, query=query)

    def format_results_only(self, results: List[QueryResult]) -> str:
        """Format just the results without the query template wrapper."""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(self.chunk_template.format(i=i, text=r.text))
        return "\n".join(parts)
