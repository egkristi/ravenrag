"""Tests for ContextFormatter."""

from ravenrag.context import ContextFormatter
from ravenrag.index import QueryResult


class TestContextFormatter:
    def test_format_default(self):
        results = [
            QueryResult(id="1", text="RAG is great", metadata={"source": "doc.txt"}, distance=0.1),
            QueryResult(id="2", text="Vectors are cool", metadata={}, distance=0.2),
        ]
        formatter = ContextFormatter()
        output = formatter.format("What is RAG?", results)

        assert "RAG is great" in output
        assert "Vectors are cool" in output
        assert "What is RAG?" in output
        assert "doc.txt" in output

    def test_format_includes_sources_section(self):
        results = [
            QueryResult(id="1", text="text", metadata={"source": "a.md", "chunk_index": 2}, distance=0.1),
        ]
        formatter = ContextFormatter()
        output = formatter.format("q", results)
        assert "Sources:" in output
        assert "a.md#chunk2" in output

    def test_format_custom_template(self):
        results = [QueryResult(id="1", text="hello", metadata={}, distance=0.1)]
        formatter = ContextFormatter(template="Q: {query}\nA: {context}")
        output = formatter.format("test", results)

        assert output.startswith("Q: test")
        assert "hello" in output

    def test_format_no_sources(self):
        results = [QueryResult(id="1", text="text", metadata={"source": "s.txt"}, distance=0.1)]
        formatter = ContextFormatter(include_sources=False)
        output = formatter.format("q", results)

        assert "s.txt" not in output

    def test_format_empty_results(self):
        formatter = ContextFormatter()
        output = formatter.format("test", [])

        assert "test" in output

    def test_format_results_only(self):
        results = [
            QueryResult(id="1", text="first", metadata={}, distance=0.1),
            QueryResult(id="2", text="second", metadata={}, distance=0.2),
        ]
        formatter = ContextFormatter()
        output = formatter.format_results_only(results)

        assert "first" in output
        assert "second" in output
        assert "[1]" in output
        assert "[2]" in output
