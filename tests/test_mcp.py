"""Tests for MCP server."""

import json
from unittest.mock import MagicMock

from ravenrag.index import QueryResult
from ravenrag.mcp_server import _handle_tool_call, _make_tools


class TestMakeTools:
    def test_returns_three_tools(self):
        index = MagicMock()
        tools = _make_tools(index)
        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"search", "get_prompt", "collection_info"}

    def test_tool_has_schema(self):
        tools = _make_tools(MagicMock())
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "inputSchema" in t


class TestHandleToolCall:
    def test_search(self):
        index = MagicMock()
        index.query.return_value = [
            QueryResult(id="d1", text="result text", metadata={"source": "a.md"}, distance=0.1),
        ]

        result = _handle_tool_call(index, "search", {"query": "test", "top_k": 3})
        parsed = json.loads(result)
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["text"] == "result text"
        index.query.assert_called_once_with("test", top_k=3)

    def test_get_prompt(self):
        index = MagicMock()
        index.query_for_prompt.return_value = "formatted prompt text"

        result = _handle_tool_call(index, "get_prompt", {"query": "explain RAG"})
        assert result == "formatted prompt text"
        index.query_for_prompt.assert_called_once_with("explain RAG", top_k=5)

    def test_collection_info(self):
        index = MagicMock()
        index.count.return_value = 42
        index.store.collection.name = "docs"

        result = _handle_tool_call(index, "collection_info", {})
        parsed = json.loads(result)
        assert parsed["documents"] == 42
        assert parsed["collection"] == "docs"

    def test_unknown_tool(self):
        index = MagicMock()
        result = _handle_tool_call(index, "nonexistent", {})
        parsed = json.loads(result)
        assert "error" in parsed
