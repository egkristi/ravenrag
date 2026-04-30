"""Tests for built-in file loaders (markdown with frontmatter)."""

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
