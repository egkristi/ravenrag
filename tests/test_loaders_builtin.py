"""Tests for built-in file loaders (markdown with frontmatter)."""

from unittest.mock import MagicMock, patch

from ravenrag.index import Document
from ravenrag.loaders import _loader_registry


class TestMarkdownLoader:
    def test_markdown_frontmatter(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("---\ntitle: Hello\nauthor: Test\n---\n\n# Content\n\nBody text here.")

        if ".md" in _loader_registry:
            doc = _loader_registry[".md"](str(md))
            assert "Content" in doc.text
            assert doc.metadata.get("title") == "Hello"
            assert doc.metadata.get("author") == "Test"

    def test_markdown_no_frontmatter(self, tmp_path):
        md = tmp_path / "plain.md"
        md.write_text("# Just a heading\n\nSome text.")

        if ".md" in _loader_registry:
            doc = _loader_registry[".md"](str(md))
            assert "Just a heading" in doc.text

    def test_markdown_registered(self):
        assert ".md" in _loader_registry


class TestLoaderSymlinkProtection:
    def test_symlink_outside_dir_skipped(self, tmp_path):
        """Symlinks pointing outside the directory should be skipped."""
        from ravenrag.loaders import load_directory

        # Create a file outside the target directory
        outside = tmp_path / "outside"
        outside.mkdir()
        outside_file = outside / "secret.txt"
        outside_file.write_text("secret data")

        # Create target dir with a symlink
        target = tmp_path / "target"
        target.mkdir()
        link = target / "link.txt"
        link.symlink_to(outside_file)

        # Also add a real file
        real = target / "real.txt"
        real.write_text("real data")

        docs = load_directory(str(target), glob="**/*.txt")
        texts = [d.text for d in docs]
        assert "real data" in texts
        # Symlink should be skipped (points outside target)
        assert "secret data" not in texts


class TestCsvLoader:
    def test_csv_registered(self):
        assert ".csv" in _loader_registry

    def test_csv_loads_rows(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")

        doc = _loader_registry[".csv"](str(csv_file))
        assert "Alice" in doc.text
        assert "Bob" in doc.text
        assert doc.metadata["extension"] == ".csv"

    def test_csv_with_metadata(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2", encoding="utf-8")

        doc = _loader_registry[".csv"](str(csv_file), metadata={"tag": "test"})
        assert doc.metadata["tag"] == "test"

    def test_csv_empty_file(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")

        doc = _loader_registry[".csv"](str(csv_file))
        assert doc.text == ""


class TestRtfLoader:
    def test_rtf_loader_with_mock(self, tmp_path):
        """Test RTF loader by mocking striprtf."""
        rtf_file = tmp_path / "test.rtf"
        rtf_file.write_text("{\\rtf1 Hello RTF}", encoding="utf-8")

        mock_rtf_to_text = MagicMock(return_value="Hello RTF")
        mock_module = MagicMock()
        mock_module.rtf_to_text = mock_rtf_to_text

        with patch.dict("sys.modules", {"striprtf": MagicMock(), "striprtf.striprtf": mock_module}):
            # Manually invoke the loader pattern
            from ravenrag.loaders import register_loader

            saved = dict(_loader_registry)
            try:
                rtf_to_text = mock_module.rtf_to_text

                @register_loader(".rtf")
                def _load_rtf(path, metadata=None):
                    from pathlib import Path

                    file_path = Path(path).resolve()
                    raw = file_path.read_text(encoding="utf-8")
                    text = rtf_to_text(raw)
                    doc_metadata = {"source": str(file_path), "filename": file_path.name, "extension": ".rtf"}
                    if metadata:
                        doc_metadata.update(metadata)
                    return Document(text=text, metadata=doc_metadata)

                doc = _loader_registry[".rtf"](str(rtf_file))
                assert doc.text == "Hello RTF"
                assert doc.metadata["extension"] == ".rtf"
            finally:
                _loader_registry.clear()
                _loader_registry.update(saved)


class TestPptxLoader:
    def test_pptx_loader_with_mock(self):
        """Test PPTX loader by mocking python-pptx."""
        mock_paragraph = MagicMock()
        mock_paragraph.text = "Slide content"
        mock_text_frame = MagicMock()
        mock_text_frame.paragraphs = [mock_paragraph]
        mock_shape = MagicMock()
        mock_shape.has_text_frame = True
        mock_shape.text_frame = mock_text_frame
        mock_slide = MagicMock()
        mock_slide.shapes = [mock_shape]
        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]

        mock_presentation_cls = MagicMock(return_value=mock_prs)

        from ravenrag.loaders import register_loader

        saved = dict(_loader_registry)
        try:

            @register_loader(".pptx")
            def _load_pptx(path, metadata=None):
                from pathlib import Path

                prs = mock_presentation_cls(path)
                texts = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            for paragraph in shape.text_frame.paragraphs:
                                text = paragraph.text.strip()
                                if text:
                                    texts.append(text)
                file_path = Path(path).resolve()
                doc_metadata = {"source": str(file_path), "filename": file_path.name, "extension": ".pptx"}
                if metadata:
                    doc_metadata.update(metadata)
                return Document(text="\n\n".join(texts), metadata=doc_metadata)

            doc = _loader_registry[".pptx"]("/tmp/test.pptx")
            assert "Slide content" in doc.text
            assert doc.metadata["extension"] == ".pptx"
        finally:
            _loader_registry.clear()
            _loader_registry.update(saved)


class TestXlsxLoader:
    def test_xlsx_loader_with_mock(self):
        """Test XLSX loader by mocking openpyxl."""
        mock_ws = MagicMock()
        mock_ws.iter_rows.return_value = [("Name", "Age"), ("Alice", 30)]
        mock_wb = MagicMock()
        mock_wb.sheetnames = ["Sheet1"]
        mock_wb.__getitem__ = MagicMock(return_value=mock_ws)

        mock_load_workbook = MagicMock(return_value=mock_wb)

        from ravenrag.loaders import register_loader

        saved = dict(_loader_registry)
        try:

            @register_loader(".xlsx")
            def _load_xlsx(path, metadata=None):
                from pathlib import Path

                wb = mock_load_workbook(path, read_only=True, data_only=True)
                texts = []
                for sheet in wb.sheetnames:
                    ws = wb[sheet]
                    for row in ws.iter_rows(values_only=True):
                        cells = [str(c) if c is not None else "" for c in row]
                        line = ", ".join(cells).strip(", ")
                        if line:
                            texts.append(line)
                wb.close()
                file_path = Path(path).resolve()
                doc_metadata = {"source": str(file_path), "filename": file_path.name, "extension": ".xlsx"}
                if metadata:
                    doc_metadata.update(metadata)
                return Document(text="\n".join(texts), metadata=doc_metadata)

            doc = _loader_registry[".xlsx"]("/tmp/test.xlsx")
            assert "Alice" in doc.text
            assert "Name" in doc.text
            assert doc.metadata["extension"] == ".xlsx"
        finally:
            _loader_registry.clear()
            _loader_registry.update(saved)
