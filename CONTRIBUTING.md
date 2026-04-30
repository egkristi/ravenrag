# Contributing to RavenRAG

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/egkristi/ravenrag.git
cd ravenrag
uv sync --all-extras
```

## Running Tests

```bash
# Fast unit tests (mocked, no model download)
uv run pytest tests/ -m "not integration" -v

# Full integration tests (downloads models on first run)
uv run pytest tests/ -m "integration" -v

# All tests with coverage
uv run pytest tests/ -v --cov=ravenrag
```

## Code Quality

```bash
# Lint
uv run ruff check ravenrag/ tests/

# Format
uv run ruff format ravenrag/ tests/
```

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Write tests for new functionality.
3. Ensure `ruff check` and `ruff format` pass.
4. Ensure all tests pass.
5. Update `CHANGELOG.md` with your changes.
6. Submit a PR with a clear description.

## License

By contributing, you agree that your contributions will be licensed under the AGPLv3 license. See [LICENSING.md](LICENSING.md) for details.
