"""Tests for CLI."""

import os

os.environ["NO_COLOR"] = "1"

from typer.testing import CliRunner

from ravenrag.cli import app

runner = CliRunner()


class TestCLI:
    def test_info_empty(self, tmp_path):
        result = runner.invoke(app, ["info", "--db", str(tmp_path / "db")])
        assert result.exit_code == 0
        assert "Documents: 0" in result.output

    def test_index_nonexistent_dir(self):
        result = runner.invoke(app, ["index", "/nonexistent/path/that/doesnt/exist"])
        assert result.exit_code != 0

    def test_query_empty_db(self, tmp_path):
        result = runner.invoke(app, ["query", "test query", "--db", str(tmp_path / "db")])
        assert "No results" in result.output

    def test_no_args_shows_help(self):
        result = runner.invoke(app)
        assert "RavenRAG" in result.output

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "index" in result.output
        assert "query" in result.output
        assert "watch" in result.output
        assert "info" in result.output
        assert "prompt" in result.output
        assert "serve" in result.output

    def test_query_has_hybrid_flag(self):
        result = runner.invoke(app, ["query", "--help"])
        assert "--hybrid" in result.output
        assert "--rerank" in result.output
        assert "--alpha" in result.output

    def test_verbose_flag(self):
        result = runner.invoke(app, ["info", "--help"])
        assert "--verbose" in result.output

    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output

    def test_export_help(self):
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "-o" in result.output

    def test_import_help(self):
        result = runner.invoke(app, ["import", "--help"])
        assert result.exit_code == 0

    def test_doctor_runs(self, tmp_path):
        result = runner.invoke(app, ["doctor", "--db", str(tmp_path / "db")])
        assert result.exit_code == 0
        assert "RavenRAG Doctor" in result.output

    def test_mcp_help(self):
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0

    def test_help_shows_new_commands(self):
        result = runner.invoke(app, ["--help"])
        assert "export" in result.output
        assert "import" in result.output
        assert "doctor" in result.output
        assert "mcp" in result.output
