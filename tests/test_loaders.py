"""Tests for document loaders."""

import pytest

from ravenrag.loaders import load_directory, load_text


class TestLoadText:
    def test_load_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world", encoding="utf-8")

        doc = load_text(str(f))
        assert doc.text == "Hello world"
        assert doc.metadata["filename"] == "test.txt"
        assert doc.metadata["source"] == str(f)

    def test_load_with_extra_metadata(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content", encoding="utf-8")

        doc = load_text(str(f), metadata={"author": "test"})
        assert doc.metadata["author"] == "test"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_text("/nonexistent/path.txt")


class TestLoadDirectory:
    def test_load_txt_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("File A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("File B", encoding="utf-8")
        (tmp_path / "c.py").write_text("# python", encoding="utf-8")

        docs = load_directory(str(tmp_path), glob="*.txt")
        assert len(docs) == 2
        texts = {d.text for d in docs}
        assert "File A" in texts
        assert "File B" in texts

    def test_recursive_glob(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested", encoding="utf-8")
        (tmp_path / "top.txt").write_text("top", encoding="utf-8")

        docs = load_directory(str(tmp_path), glob="**/*.txt")
        assert len(docs) == 2

    def test_not_a_directory(self):
        with pytest.raises(NotADirectoryError):
            load_directory("/nonexistent/dir")

    def test_skips_binary_files(self, tmp_path):
        (tmp_path / "binary.txt").write_bytes(b"\x80\x81\x82\x83" * 100)
        (tmp_path / "good.txt").write_text("readable", encoding="utf-8")

        docs = load_directory(str(tmp_path), glob="*.txt")
        assert len(docs) == 1
        assert docs[0].text == "readable"
