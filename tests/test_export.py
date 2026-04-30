"""Tests for export/import functionality."""

import io
import json
from unittest.mock import MagicMock

from ravenrag.export import export_index, import_index


class TestExportIndex:
    def test_export_writes_jsonl(self):
        index = MagicMock()
        index.store.get_all.return_value = {
            "ids": ["d1", "d2"],
            "documents": ["text one", "text two"],
            "metadatas": [{"source": "a.txt"}, {"source": "b.txt"}],
        }

        output = io.StringIO()
        count = export_index(index, output)

        assert count == 2
        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 2

        record = json.loads(lines[0])
        assert record["id"] == "d1"
        assert record["text"] == "text one"
        assert record["metadata"]["source"] == "a.txt"

    def test_export_empty_index(self):
        index = MagicMock()
        index.store.get_all.return_value = {"ids": [], "documents": [], "metadatas": []}

        output = io.StringIO()
        count = export_index(index, output)
        assert count == 0
        assert output.getvalue() == ""

    def test_export_no_metadata(self):
        index = MagicMock()
        index.store.get_all.return_value = {
            "ids": ["d1"],
            "documents": ["text"],
            "metadatas": None,
        }

        output = io.StringIO()
        count = export_index(index, output)
        assert count == 1
        record = json.loads(output.getvalue().strip())
        assert record["metadata"] == {}


class TestImportIndex:
    def test_import_from_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "backup.jsonl"
        records = [
            {"id": "d1", "text": "text one", "metadata": {"source": "a.txt"}},
            {"id": "d2", "text": "text two", "metadata": {}},
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

        index = MagicMock()
        count = import_index(index, str(jsonl_file))

        assert count == 2
        index.add.assert_called_once()
        docs = index.add.call_args[0][0]
        assert len(docs) == 2
        assert docs[0].text == "text one"

    def test_import_skips_invalid_json(self, tmp_path):
        jsonl_file = tmp_path / "bad.jsonl"
        jsonl_file.write_text('{"text": "good"}\nnot json\n{"text": "also good"}\n', encoding="utf-8")

        index = MagicMock()
        count = import_index(index, str(jsonl_file))
        assert count == 2

    def test_import_skips_missing_text(self, tmp_path):
        jsonl_file = tmp_path / "notext.jsonl"
        jsonl_file.write_text('{"id": "d1"}\n{"text": "valid"}\n', encoding="utf-8")

        index = MagicMock()
        count = import_index(index, str(jsonl_file))
        assert count == 1

    def test_import_file_not_found(self):
        import pytest

        index = MagicMock()
        with pytest.raises(FileNotFoundError):
            import_index(index, "/nonexistent/backup.jsonl")

    def test_import_empty_file(self, tmp_path):
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("", encoding="utf-8")

        index = MagicMock()
        count = import_index(index, str(jsonl_file))
        assert count == 0
        index.add.assert_not_called()
