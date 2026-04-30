"""Tests for watcher module."""

from unittest.mock import MagicMock, patch

from ravenrag.watcher import _file_doc_id


class TestFileDocId:
    def test_deterministic(self, tmp_path):
        p = tmp_path / "test.txt"
        assert _file_doc_id(p) == _file_doc_id(p)

    def test_different_files_different_ids(self, tmp_path):
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        assert _file_doc_id(p1) != _file_doc_id(p2)

    def test_returns_sha256_hex(self, tmp_path):
        p = tmp_path / "test.txt"
        result = _file_doc_id(p)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestWatchDirectory:
    def test_import_error_without_watchfiles(self, tmp_path):
        import pytest

        from ravenrag.watcher import watch_directory

        index = MagicMock()
        with patch.dict("sys.modules", {"watchfiles": None}):
            # Force re-import failure
            with pytest.raises(ImportError, match="watchfiles"):
                watch_directory(str(tmp_path), index)

    def test_not_a_directory_raises(self, tmp_path):
        import pytest

        from ravenrag.watcher import watch_directory

        index = MagicMock()
        with pytest.raises(NotADirectoryError):
            watch_directory("/nonexistent/path/that/doesnt/exist", index)
