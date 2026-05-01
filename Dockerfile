# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────
# RavenRAG — Lightweight, local-first RAG server
# ──────────────────────────────────────────────
# Build:  docker build -t ravenrag .
# Run:    docker run -p 8484:8484 -v ravenrag-data:/data ravenrag
# ──────────────────────────────────────────────

FROM python:3.12-slim AS builder

# Avoid interactive prompts and bytecode files
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency metadata first (cache-friendly layer)
COPY pyproject.toml README.md LICENSE ./
COPY setup.py ./
COPY ravenrag/ ravenrag/

# Install the package with all extras into a virtual env
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python ".[all]"

# Pre-download the default embedding model so first query is fast
RUN /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ──────────────────────────────────────────────
# Runtime stage — smaller final image
# ──────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd --gid 1000 raven && \
    useradd --uid 1000 --gid raven --create-home raven

WORKDIR /app

# Copy the virtual env and model cache from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /root/.cache/huggingface /home/raven/.cache/huggingface

# Fix ownership of model cache
RUN chown -R raven:raven /home/raven/.cache

# Copy application code
COPY ravenrag/ ravenrag/
COPY pyproject.toml README.md LICENSE ./

# Put venv on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Default data directory (mount a volume here for persistence)
ENV RAVENRAG_DB=/data
RUN mkdir -p /data && chown raven:raven /data
VOLUME ["/data"]

# Switch to non-root user
USER raven

# Expose the API server port
EXPOSE 8484

# Health check using the built-in /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8484/health')" || exit 1

# Default: start the HTTP API server, bind to all interfaces
ENTRYPOINT ["raven"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8484"]
