"""Tests for the Pipeline API."""

from unittest.mock import MagicMock

from ravenrag.index import Document, QueryResult
from ravenrag.pipeline import Pipeline


class TestPipeline:
    def test_run_basic(self):
        docs = [Document(text="hello world", metadata={"source": "test"})]
        loader = MagicMock(return_value=docs)
        index = MagicMock()
        index.count.return_value = 1

        pipe = Pipeline(loader=loader, index=index)
        stats = pipe.run("./test")

        loader.assert_called_once_with("./test")
        index.add.assert_called_once_with(docs)
        assert stats["loaded"] == 1
        assert stats["indexed"] == 1
        assert stats["errors"] == 0

    def test_run_with_splitter(self):
        docs = [Document(text="long text", metadata={"source": "test"})]
        chunks = [
            Document(text="chunk 1", metadata={"source": "test", "chunk_index": 0}),
            Document(text="chunk 2", metadata={"source": "test", "chunk_index": 1}),
        ]
        loader = MagicMock(return_value=docs)
        splitter = MagicMock()
        splitter.split_documents.return_value = chunks
        index = MagicMock()

        pipe = Pipeline(loader=loader, index=index, splitter=splitter)
        stats = pipe.run("./test")

        splitter.split_documents.assert_called_once_with(docs)
        index.add.assert_called_once_with(chunks)
        assert stats["loaded"] == 1
        assert stats["chunked"] == 2
        assert stats["indexed"] == 2

    def test_run_error_skip(self):
        loader = MagicMock(side_effect=RuntimeError("load failed"))
        index = MagicMock()

        pipe = Pipeline(loader=loader, index=index, on_error="skip")
        stats = pipe.run("./test")

        assert stats["errors"] == 1
        assert stats["loaded"] == 0
        index.add.assert_not_called()

    def test_run_error_raise(self):
        loader = MagicMock(side_effect=RuntimeError("load failed"))
        index = MagicMock()

        pipe = Pipeline(loader=loader, index=index, on_error="raise")
        try:
            pipe.run("./test")
            assert False, "Should have raised"
        except RuntimeError:
            pass

    def test_query(self):
        loader = MagicMock()
        index = MagicMock()
        index.query.return_value = [QueryResult(id="1", text="result", metadata={}, distance=0.1)]

        pipe = Pipeline(loader=loader, index=index)
        results = pipe.query("test query", top_k=3)

        index.query.assert_called_once_with("test query", top_k=3)
        assert len(results) == 1

    def test_query_stream(self):
        loader = MagicMock()
        index = MagicMock()
        index.query.return_value = [
            QueryResult(id="1", text="r1", metadata={}, distance=0.1),
            QueryResult(id="2", text="r2", metadata={}, distance=0.2),
        ]

        pipe = Pipeline(loader=loader, index=index)
        streamed = list(pipe.query_stream("test"))

        assert len(streamed) == 2
        assert streamed[0].id == "1"

    def test_stats_property(self):
        pipe = Pipeline(loader=MagicMock(return_value=[]), index=MagicMock())
        pipe.run("./test")
        stats = pipe.stats
        assert "loaded" in stats
        assert "errors" in stats
