"""
CLI: Command-line interface for RavenRAG.
"""

from __future__ import annotations

import logging
from typing import Optional

import typer

app = typer.Typer(
    name="raven",
    help="RavenRAG — Index your docs in one command, query them from anywhere.",
    no_args_is_help=True,
)

_verbose_option = typer.Option(False, "--verbose", "-v", help="Enable debug logging")
_db_option = typer.Option(None, "--db", help="Database directory (overrides config)")
_collection_option = typer.Option(None, "--collection", help="Collection name (overrides config)")
_model_option = typer.Option(None, "--model", help="Embedding model name (overrides config)")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _get_config():
    from .config import load_config

    return load_config()


@app.command()
def index(
    path: str = typer.Argument(..., help="Directory to index"),
    glob: str = typer.Option(None, help="Glob pattern for files (overrides config)"),
    persist_dir: Optional[str] = _db_option,
    chunk_size: Optional[int] = typer.Option(None, help="Chunk size in characters"),
    chunk_overlap: Optional[int] = typer.Option(None, help="Chunk overlap in characters"),
    collection: Optional[str] = _collection_option,
    model: Optional[str] = _model_option,
    verbose: bool = _verbose_option,
) -> None:
    """Index documents from a directory."""
    _setup_logging(verbose)
    from . import DocumentIndex, TextSplitter, load_directory

    cfg = _get_config()
    db = persist_dir or cfg.index.persist_dir
    col = collection or cfg.index.collection
    mdl = model or cfg.index.model
    g = glob or cfg.index.glob
    cs = chunk_size or cfg.index.chunk_size
    co = chunk_overlap or cfg.index.chunk_overlap

    typer.echo(f"Loading files from {path} ({g})...")
    docs = load_directory(path, glob=g)

    if not docs:
        typer.echo("No documents found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(docs)} files.")

    splitter = TextSplitter(chunk_size=cs, chunk_overlap=co)
    chunks = splitter.split_documents(docs)
    typer.echo(f"Split into {len(chunks)} chunks.")

    idx = DocumentIndex(persist_dir=db, collection_name=col, embedding_model=mdl)
    idx.add(chunks)
    typer.echo(f"Indexed {idx.count()} documents in {db}")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Search query"),
    persist_dir: Optional[str] = _db_option,
    top_k: Optional[int] = typer.Option(None, "-k", help="Number of results"),
    collection: Optional[str] = _collection_option,
    model: Optional[str] = _model_option,
    rerank: Optional[bool] = typer.Option(None, "--rerank", help="Rerank with cross-encoder"),
    hybrid: Optional[bool] = typer.Option(None, "--hybrid", help="Use hybrid search (vector + BM25)"),
    alpha: Optional[float] = typer.Option(None, "--alpha", help="Hybrid alpha: 1.0=vector, 0.0=BM25"),
    verbose: bool = _verbose_option,
) -> None:
    """Query the index for similar documents."""
    _setup_logging(verbose)
    from . import DocumentIndex

    cfg = _get_config()
    db = persist_dir or cfg.index.persist_dir
    col = collection or cfg.index.collection
    mdl = model or cfg.index.model
    k = top_k if top_k is not None else cfg.search.top_k
    use_hybrid = hybrid if hybrid is not None else cfg.search.hybrid
    use_rerank = rerank if rerank is not None else cfg.search.rerank
    a = alpha if alpha is not None else cfg.search.alpha

    idx = DocumentIndex(persist_dir=db, collection_name=col, embedding_model=mdl)

    if use_hybrid:
        results = idx.hybrid_query(query_text, top_k=k, alpha=a)
    else:
        results = idx.query(query_text, top_k=k, rerank=use_rerank)

    if not results:
        typer.echo("No results found.")
        raise typer.Exit(1)

    for i, r in enumerate(results, 1):
        typer.echo(f"\n[{i}] (distance: {r.distance:.4f})")
        typer.echo(f"    Source: {r.citation}")
        typer.echo(f"    {r.text[:200]}")


@app.command()
def prompt(
    query_text: str = typer.Argument(..., help="Search query"),
    persist_dir: Optional[str] = _db_option,
    top_k: Optional[int] = typer.Option(None, "-k", help="Number of results"),
    collection: Optional[str] = _collection_option,
    model: Optional[str] = _model_option,
    verbose: bool = _verbose_option,
) -> None:
    """Query and output a formatted LLM prompt with context."""
    _setup_logging(verbose)
    from . import DocumentIndex

    cfg = _get_config()
    db = persist_dir or cfg.index.persist_dir
    col = collection or cfg.index.collection
    mdl = model or cfg.index.model
    k = top_k if top_k is not None else cfg.search.top_k

    idx = DocumentIndex(persist_dir=db, collection_name=col, embedding_model=mdl)
    output = idx.query_for_prompt(query_text, top_k=k)
    typer.echo(output)


@app.command()
def serve(
    persist_dir: Optional[str] = _db_option,
    collection: Optional[str] = _collection_option,
    model: Optional[str] = _model_option,
    host: Optional[str] = typer.Option(None, help="Bind address"),
    port: Optional[int] = typer.Option(None, help="Bind port"),
    verbose: bool = _verbose_option,
) -> None:
    """Start the RavenRAG API server."""
    _setup_logging(verbose)
    from .server import serve as _serve

    cfg = _get_config()
    _serve(
        persist_dir=persist_dir or cfg.index.persist_dir,
        collection_name=collection or cfg.index.collection,
        embedding_model=model or cfg.index.model,
        host=host or cfg.server.host,
        port=port or cfg.server.port,
    )


@app.command()
def watch(
    path: str = typer.Argument(..., help="Directory to watch"),
    persist_dir: Optional[str] = _db_option,
    extensions: Optional[str] = typer.Option(None, help="Comma-separated file extensions"),
    chunk_size: Optional[int] = typer.Option(None, help="Chunk size"),
    chunk_overlap: Optional[int] = typer.Option(None, help="Chunk overlap"),
    collection: Optional[str] = _collection_option,
    model: Optional[str] = _model_option,
    verbose: bool = _verbose_option,
) -> None:
    """Watch a directory and auto-reindex on changes."""
    _setup_logging(verbose)
    from . import DocumentIndex, TextSplitter
    from .watcher import watch_directory

    cfg = _get_config()
    db = persist_dir or cfg.index.persist_dir
    col = collection or cfg.index.collection
    mdl = model or cfg.index.model
    cs = chunk_size or cfg.index.chunk_size
    co = chunk_overlap or cfg.index.chunk_overlap

    if extensions:
        exts = [e.strip() if e.startswith(".") else f".{e.strip()}" for e in extensions.split(",")]
    else:
        exts = cfg.watch_extensions

    idx = DocumentIndex(persist_dir=db, collection_name=col, embedding_model=mdl)
    splitter = TextSplitter(chunk_size=cs, chunk_overlap=co)

    typer.echo(f"Watching {path} for changes ({', '.join(exts)})...")
    typer.echo("Press Ctrl+C to stop.")

    watch_directory(path, idx, extensions=exts, splitter=splitter)


@app.command()
def info(
    persist_dir: Optional[str] = _db_option,
    collection: Optional[str] = _collection_option,
    verbose: bool = _verbose_option,
) -> None:
    """Show index statistics."""
    _setup_logging(verbose)
    from . import DocumentIndex

    cfg = _get_config()
    db = persist_dir or cfg.index.persist_dir
    col = collection or cfg.index.collection

    idx = DocumentIndex(persist_dir=db, collection_name=col)
    count = idx.count()
    typer.echo(f"Database: {db}")
    typer.echo(f"Collection: {col}")
    typer.echo(f"Documents: {count}")
