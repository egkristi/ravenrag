"""Tests for CLI."""

import re

from typer.testing import CliRunner

from ravenrag.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[^m]*m")


def _clean(text: str) -> str:
    """Strip ANSI escape codes from text."""
    return _ANSI_RE.sub("", text)


class TestCLI:
    def test_info_empty(self, tmp_path):
        result = runner.invoke(app, ["info", "--db", str(tmp_path / "db")])
        assert result.exit_code == 0
        assert "Documents: 0" in _clean(result.output)

    def test_index_nonexistent_dir(self):
        result = runner.invoke(app, ["index", "/nonexistent/path/that/doesnt/exist"])
        assert result.exit_code != 0

    def test_query_empty_db(self, tmp_path):
        result = runner.invoke(app, ["query", "test query", "--db", str(tmp_path / "db")])
        assert "No results" in _clean(result.output)

    def test_no_args_shows_help(self):
        result = runner.invoke(app)
        assert "RavenRAG" in _clean(result.output)

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = _clean(result.output)
        assert "index" in output
        assert "query" in output
        assert "watch" in output
        assert "info" in output
        assert "prompt" in output
        assert "serve" in output

    def test_query_has_hybrid_flag(self):
        result = runner.invoke(app, ["query", "--help"])
        output = _clean(result.output)
        assert "--hybrid" in output
        assert "--rerank" in output
        assert "--alpha" in output

    def test_verbose_flag(self):
        result = runner.invoke(app, ["info", "--help"])
        assert "--verbose" in _clean(result.output)

    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        output = _clean(result.output)
        assert "--host" in output
        assert "--port" in output

    def test_export_help(self):
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "-o" in _clean(result.output)

    def test_import_help(self):
        result = runner.invoke(app, ["import", "--help"])
        assert result.exit_code == 0

    def test_doctor_runs(self, tmp_path):
        result = runner.invoke(app, ["doctor", "--db", str(tmp_path / "db")])
        assert result.exit_code == 0
        assert "RavenRAG Doctor" in _clean(result.output)

    def test_mcp_help(self):
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0

    def test_help_shows_new_commands(self):
        result = runner.invoke(app, ["--help"])
        output = _clean(result.output)
        assert "export" in output
        assert "import" in output
        assert "doctor" in output
        assert "mcp" in output
