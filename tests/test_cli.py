"""Tests for CLI."""

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
