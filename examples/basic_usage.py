"""
Basic usage example for RavenRAG.
"""

from ravenrag import DocumentIndex, Document, TextSplitter, load_directory


def main():
    # Create a local index
    index = DocumentIndex(persist_dir="./example_db")

    # --- Manual documents with metadata ---
    documents = [
        Document(
            "RAG (Retrieval-Augmented Generation) combines retrieval with generation to produce more accurate answers.",
            metadata={"source": "wikipedia", "topic": "ai"},
        ),
        Document(
            "ChromaDB is an open-source embedding database that makes it easy to build LLM apps.",
            metadata={"source": "chromadb", "topic": "database"},
        ),
        Document(
            "Sentence-transformers maps sentences to dense vector representations.",
            metadata={"source": "huggingface", "topic": "nlp"},
        ),
        Document(
            "Vector search finds the most similar items in a high-dimensional space.",
            metadata={"source": "tutorial", "topic": "search"},
        ),
    ]

    print(f"Indexing {len(documents)} documents...")
    index.add(documents)
    print(f"Total indexed: {index.count()}")

    # --- Query ---
    queries = [
        "What is RAG?",
        "Tell me about vector databases",
        "How do embeddings work?",
    ]

    for q in queries:
        print(f"\n🔍 Query: {q}")
        results = index.query(q, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['distance']:.4f}] {r['text'][:80]}...")

    # --- Metadata filtering ---
    print("\n🏷️  Filtering by topic='ai':")
    results = index.query("tell me something", top_k=2, where={"topic": "ai"})
    for r in results:
        print(f"  [{r['distance']:.4f}] {r['text'][:80]}...")

    # --- Chunking example ---
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    long_doc = Document("This is a very long document. " * 20, metadata={"source": "demo"})
    chunks = splitter.split_documents([long_doc])
    print(f"\n✂️  Split 1 document into {len(chunks)} chunks")

    print(f"\n✅ Done. Database persisted to ./example_db")


if __name__ == "__main__":
    main()
