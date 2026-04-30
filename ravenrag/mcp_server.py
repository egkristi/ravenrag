"""
MCP Server: Expose RavenRAG as a Model Context Protocol server.

This lets AI assistants (Claude, Copilot, Cursor, etc.) query your
local documents directly via the MCP standard.

Uses the ``mcp`` package (stdio transport).
Requires: ``pip install 'ravenrag[mcp]'``

Start with::

    raven mcp
    # or
    python -m ravenrag.mcp_server
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _make_tools(index: Any) -> list[dict]:
    """Return MCP tool definitions."""
    return [
        {
            "name": "search",
            "description": "Search the RavenRAG document index for relevant passages.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "top_k": {"type": "integer", "description": "Number of results.", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_prompt",
            "description": "Search and return a formatted LLM prompt with context from indexed documents.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question to answer."},
                    "top_k": {"type": "integer", "description": "Number of context chunks.", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "collection_info",
            "description": "Get statistics about the indexed document collection.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


def _handle_tool_call(index: Any, name: str, arguments: dict) -> str:
    """Execute an MCP tool call and return the result as a string."""
    if name == "search":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        results = index.query(query, top_k=top_k)
        items = []
        for r in results:
            items.append({"text": r.text, "citation": r.citation, "distance": r.distance})
        return json.dumps({"results": items}, indent=2)

    elif name == "get_prompt":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        prompt = index.query_for_prompt(query, top_k=top_k)
        return prompt

    elif name == "collection_info":
        return json.dumps(
            {
                "documents": index.count(),
                "collection": index.store.collection.name,
            },
            indent=2,
        )

    return json.dumps({"error": f"Unknown tool: {name}"})


def run_stdio_server(
    persist_dir: str = "./ravenrag_db",
    collection_name: str = "documents",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> None:
    """Run the MCP server using stdio transport (JSON-RPC over stdin/stdout).

    This implements the MCP protocol directly using stdin/stdout,
    without requiring the ``mcp`` Python package.
    """
    from .index import DocumentIndex

    index = DocumentIndex(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    tools = _make_tools(index)

    logger.info("RavenRAG MCP server starting (stdio)")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _respond({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None})
            continue

        req_id = request.get("id")
        method = request.get("method", "")

        if method == "initialize":
            _respond(
                {
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "ravenrag", "version": "0.5.0"},
                    },
                    "id": req_id,
                }
            )
        elif method == "notifications/initialized":
            pass  # Client acknowledgment, no response needed
        elif method == "tools/list":
            _respond({"jsonrpc": "2.0", "result": {"tools": tools}, "id": req_id})
        elif method == "tools/call":
            params = request.get("params", {})
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            try:
                result_text = _handle_tool_call(index, name, arguments)
                _respond(
                    {
                        "jsonrpc": "2.0",
                        "result": {"content": [{"type": "text", "text": result_text}]},
                        "id": req_id,
                    }
                )
            except Exception as e:
                _respond(
                    {
                        "jsonrpc": "2.0",
                        "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True},
                        "id": req_id,
                    }
                )
        else:
            _respond(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": req_id,
                }
            )


def _respond(msg: dict) -> None:
    """Write a JSON-RPC response to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
