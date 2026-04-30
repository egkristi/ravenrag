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

    # Incremental re-indexing: skip unchanged files
    from .fingerprint import FingerprintStore

    fp_store = FingerprintStore(db)
    from pathlib import Path as _Path

    file_paths = [_Path(d.metadata.get("source", "")) for d in docs if d.metadata.get("source")]
    changed, unchanged, deleted = fp_store.diff(file_paths)
    changed_sources = {str(p.resolve()) for p in changed}

    if unchanged:
        typer.echo(f"Skipping {len(unchanged)} unchanged files.")
    if deleted:
        typer.echo(f"Removing {len(deleted)} deleted files from index.")
        idx_del = DocumentIndex(persist_dir=db, collection_name=col, embedding_model=mdl)
        for key in deleted:
            # Remove documents whose source matches this key
            all_data = idx_del.store.get_all()
            for i, meta in enumerate(all_data.get("metadatas") or []):
                if meta and str(meta.get("source", "")) == key:
                    try:
                        idx_del.delete(all_data["ids"][i])
                    except Exception:
                        pass
            fp_store.remove(key)

    docs = [d for d in docs if d.metadata.get("source") in changed_sources or not d.metadata.get("source")]

    if not docs and not deleted:
        typer.echo("Everything up to date.")
        fp_store.save()
        return

    typer.echo(f"Found {len(docs)} new/changed files.")

    splitter = TextSplitter(chunk_size=cs, chunk_overlap=co)
    chunks = splitter.split_documents(docs)
    typer.echo(f"Split into {len(chunks)} chunks.")

    idx = DocumentIndex(persist_dir=db, collection_name=col, embedding_model=mdl)
    idx.add(chunks)

    # Update fingerprints for indexed files
    for p in changed:
        fp_store.update(p)
    fp_store.save()

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
        api_key=cfg.server.api_key or None,
        cors_origin=cfg.server.cors_origin or None,
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


@app.command("export")
def export_cmd(
    persist_dir: Optional[str] = _db_option,
    collection: Optional[str] = _collection_option,
    model: Optional[str] = _model_option,
    output: Optional[str] = typer.Option(None, "-o", help="Output file (default: stdout)"),
    verbose: bool = _verbose_option,
) -> None:
    """Export all documents as JSONL."""
    _setup_logging(verbose)
    import sys

    from . import DocumentIndex
    from .export import export_index

    cfg = _get_config()
    db = persist_dir or cfg.index.persist_dir
    col = collection or cfg.index.collection
    mdl = model or cfg.index.model

    idx = DocumentIndex(persist_dir=db, collection_name=col, embedding_model=mdl)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            n = export_index(idx, f)
        typer.echo(f"Exported {n} documents to {output}", err=True)
    else:
        n = export_index(idx, sys.stdout)
        typer.echo(f"Exported {n} documents", err=True)


@app.command("import")
def import_cmd(
    input_file: str = typer.Argument(..., help="JSONL file to import"),
    persist_dir: Optional[str] = _db_option,
    collection: Optional[str] = _collection_option,
    model: Optional[str] = _model_option,
    verbose: bool = _verbose_option,
) -> None:
    """Import documents from a JSONL file."""
    _setup_logging(verbose)
    from . import DocumentIndex
    from .export import import_index

    cfg = _get_config()
    db = persist_dir or cfg.index.persist_dir
    col = collection or cfg.index.collection
    mdl = model or cfg.index.model

    idx = DocumentIndex(persist_dir=db, collection_name=col, embedding_model=mdl)
    n = import_index(idx, input_file)
    typer.echo(f"Imported {n} documents. Total: {idx.count()}")


@app.command()
def doctor(
    persist_dir: Optional[str] = _db_option,
    collection: Optional[str] = _collection_option,
    verbose: bool = _verbose_option,
) -> None:
    """Diagnose your RavenRAG setup."""
    _setup_logging(verbose)
    from .config import load_config

    cfg = load_config()
    db = persist_dir or cfg.index.persist_dir
    col = collection or cfg.index.collection

    typer.echo("🐦‍⬛ RavenRAG Doctor\n")

    # Check config
    typer.echo(f"  Config: persist_dir={cfg.index.persist_dir}, model={cfg.index.model}")

    # Check database
    from pathlib import Path

    db_path = Path(db)
    if db_path.exists():
        typer.echo(f"  Database: {db} ✓ (exists)")
    else:
        typer.echo(f"  Database: {db} ✗ (not found)")

    # Check ChromaDB
    try:
        from . import DocumentIndex

        idx = DocumentIndex(persist_dir=db, collection_name=col)
        typer.echo(f"  ChromaDB: {idx.count()} documents in '{col}' ✓")
    except Exception as e:
        typer.echo(f"  ChromaDB: ✗ ({e})")

    # Check embedding model
    try:
        from .embed import Embedder

        emb = Embedder(cfg.index.model)
        emb.encode(["test"])
        typer.echo(f"  Embedder: {cfg.index.model} ✓")
    except Exception as e:
        typer.echo(f"  Embedder: {cfg.index.model} ✗ ({e})")

    # Check optional deps
    optional = [
        ("rank_bm25", "hybrid search"),
        ("watchfiles", "watch mode"),
        ("transformers", "token splitting"),
    ]
    for pkg, feature in optional:
        try:
            __import__(pkg)
            typer.echo(f"  {feature}: ✓")
        except ImportError:
            typer.echo(f"  {feature}: ✗ (not installed)")


@app.command()
def mcp(
    persist_dir: Optional[str] = _db_option,
    collection: Optional[str] = _collection_option,
    model: Optional[str] = _model_option,
    verbose: bool = _verbose_option,
) -> None:
    """Start the MCP (Model Context Protocol) server for AI assistants."""
    _setup_logging(verbose)
    from .mcp_server import run_stdio_server

    cfg = _get_config()
    run_stdio_server(
        persist_dir=persist_dir or cfg.index.persist_dir,
        collection_name=collection or cfg.index.collection,
        embedding_model=model or cfg.index.model,
    )
