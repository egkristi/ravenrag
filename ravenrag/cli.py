"""
CLI: Command-line interface for RavenRAG.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="raven",
    help="RavenRAG — Index your docs in one command, query them from anywhere.",
    no_args_is_help=True,
)


@app.command()
def index(
    path: str = typer.Argument(..., help="Directory to index"),
    glob: str = typer.Option("**/*.txt", help="Glob pattern for files"),
    persist_dir: str = typer.Option("./ravenrag_db", "--db", help="Database directory"),
    chunk_size: int = typer.Option(512, help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(64, help="Chunk overlap in characters"),
    collection: str = typer.Option("documents", help="Collection name"),
    model: str = typer.Option("all-MiniLM-L6-v2", help="Embedding model name"),
) -> None:
    """Index documents from a directory."""
    from . import DocumentIndex, TextSplitter, load_directory

    typer.echo(f"Loading files from {path} ({glob})...")
    docs = load_directory(path, glob=glob)

    if not docs:
        typer.echo("No documents found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(docs)} files.")

    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    typer.echo(f"Split into {len(chunks)} chunks.")

    idx = DocumentIndex(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_model=model,
    )
    idx.add(chunks)
    typer.echo(f"Indexed {idx.count()} documents in {persist_dir}")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Search query"),
    persist_dir: str = typer.Option("./ravenrag_db", "--db", help="Database directory"),
    top_k: int = typer.Option(5, "-k", help="Number of results"),
    collection: str = typer.Option("documents", help="Collection name"),
    model: str = typer.Option("all-MiniLM-L6-v2", help="Embedding model name"),
) -> None:
    """Query the index for similar documents."""
    from . import DocumentIndex

    idx = DocumentIndex(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_model=model,
    )
    results = idx.query(query_text, top_k=top_k)

    if not results:
        typer.echo("No results found.")
        raise typer.Exit(1)

    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", r.metadata.get("filename", ""))
        typer.echo(f"\n[{i}] (distance: {r.distance:.4f})")
        if source:
            typer.echo(f"    Source: {source}")
        typer.echo(f"    {r.text[:200]}")


@app.command()
def prompt(
    query_text: str = typer.Argument(..., help="Search query"),
    persist_dir: str = typer.Option("./ravenrag_db", "--db", help="Database directory"),
    top_k: int = typer.Option(5, "-k", help="Number of results"),
    collection: str = typer.Option("documents", help="Collection name"),
    model: str = typer.Option("all-MiniLM-L6-v2", help="Embedding model name"),
) -> None:
    """Query and output a formatted LLM prompt with context."""
    from . import DocumentIndex

    idx = DocumentIndex(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_model=model,
    )
    output = idx.query_for_prompt(query_text, top_k=top_k)
    typer.echo(output)


@app.command()
def watch(
    path: str = typer.Argument(..., help="Directory to watch"),
    persist_dir: str = typer.Option("./ravenrag_db", "--db", help="Database directory"),
    extensions: str = typer.Option(".txt,.md,.py", help="Comma-separated file extensions"),
    chunk_size: int = typer.Option(512, help="Chunk size"),
    chunk_overlap: int = typer.Option(64, help="Chunk overlap"),
    collection: str = typer.Option("documents", help="Collection name"),
    model: str = typer.Option("all-MiniLM-L6-v2", help="Embedding model name"),
) -> None:
    """Watch a directory and auto-reindex on changes."""
    from . import DocumentIndex, TextSplitter
    from .watcher import watch_directory

    exts = [e.strip() if e.startswith(".") else f".{e.strip()}" for e in extensions.split(",")]

    idx = DocumentIndex(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_model=model,
    )
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    typer.echo(f"Watching {path} for changes ({', '.join(exts)})...")
    typer.echo("Press Ctrl+C to stop.")

    watch_directory(path, idx, extensions=exts, splitter=splitter)


@app.command()
def info(
    persist_dir: str = typer.Option("./ravenrag_db", "--db", help="Database directory"),
    collection: str = typer.Option("documents", help="Collection name"),
) -> None:
    """Show index statistics."""
    from . import DocumentIndex

    idx = DocumentIndex(persist_dir=persist_dir, collection_name=collection)
    count = idx.count()
    typer.echo(f"Database: {persist_dir}")
    typer.echo(f"Collection: {collection}")
    typer.echo(f"Documents: {count}")
