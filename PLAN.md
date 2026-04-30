# RavenRAG — Audit & Plan

> Dybde-analyse utført 30. april 2026 mot `main` (commit `ef51ac4`).

---

## Status quo

| Metrikk | Verdi |
|---------|-------|
| Versjon | 0.4.0 |
| Python | ≥ 3.10 |
| Tester | 94 (alle grønn) |
| Moduler | 12 source-filer, 9 test-filer |
| Backends | sentence-transformers, Ollama, OpenAI-compatible, vLLM |
| Ekstra | CLI, HTTP-server, hybrid search, reranking, config, plugin loaders, watcher |

Prosjektet har solid grunnmur. Det som skiller «fungerer» fra «virkelig bra» er robusthet, developer experience, og noen killer-features ingen andre lette RAG-libs har.

---

## Del 1 — Bugs & kritiske funn

### 1.1 Ruff target-version mismatch
`pyproject.toml` sier `requires-python = ">=3.10"`, men `[tool.ruff]` har `target-version = "py39"`.
**Fix:** Sett `target-version = "py310"`.

### 1.2 Ingen timeout på HTTP-kall (embed.py)
`urllib.request.urlopen()` i `OllamaBackend`, `OpenAIBackend` og `VLLMBackend` har ingen `timeout`.
En treg/død server henger prosessen for alltid.
**Fix:** Legg til `timeout=30` (konfigurerbar) på alle `urlopen()`-kall.

### 1.3 Server godtar ubegrenset request-body
`_read_json()` i `server.py` leser hele `Content-Length` rett i minne uten limit.
En enkel `curl -d @/dev/urandom` tar ned serveren.
**Fix:** Maks 10 MB body, returner 413 Payload Too Large.

### 1.4 top_k-validering mangler i server
POST `/query` videresender `top_k` direkte til ChromaDB uten sjekk.
**Fix:** Clamp til `1 ≤ top_k ≤ 1000`.

### 1.5 Watcher håndterer ikke sletting
Når en fil slettes logges bare en warning — dokumentet blir i indeksen for alltid.
**Fix:** Kall `index.delete()` ved fil-sletting.

### 1.6 Watcher mangler debounce
100 filendringer på 1 sekund = 100 re-indekseringer.
**Fix:** Samle endringer i et vindu (f.eks. 500 ms) og kjør én gang.

### 1.7 get_all() i hybrid search skalerer ikke
`HybridSearcher` kaller `store.get_all()` som henter *alle* dokumenter i minne per query.
Med 100k+ docs er dette en showstopper.
**Fix:** Paginert henting, eller pre-bygg BM25-indeks som caches.

### 1.8 Document ID-kollisjoner
To dokumenter med identisk tekst får samme SHA256-ID → `upsert()` overskriver stille.
**Fix:** Warn ved kollisjon, eller inkluder metadata i hash.

---

## Del 2 — Sikkerhetshuller

| # | Problem | Alvorlighet | Fix |
|---|---------|-------------|-----|
| S1 | Ingen auth på HTTP-server | Kritisk | API-key header eller Bearer token |
| S2 | Ingen CORS-headers | Høy | Konfigurerbar `Access-Control-Allow-Origin` |
| S3 | Ingen request-størrelse-limit | Høy | 10 MB maks (1.3 over) |
| S4 | Path traversal i `load_directory` | Middels | Valider at resolvede stier er innenfor root |
| S5 | API-nøkkel kan lekke i feilmeldinger | Middels | Maskér nøkkel i exception-meldinger |
| S6 | Stats-endepunkt eksponerer internals | Lav | Fjern `persist_dir`, filtrer metadata |

---

## Del 3 — Test-gap

| Mangler | Prioritet |
|---------|-----------|
| **test_store.py** — VectorStore er helt utestet | Kritisk |
| **test_watcher.py** — Watcher er utestet | Høy |
| CLI happy-path (faktisk indexering + query) | Høy |
| Server: fulle HTTP-request-tester | Høy |
| Config: ugyldige verdier, edge cases | Middels |
| Embed: model download failure | Middels |
| Hybrid: metadata where-filtering | Middels |

Mål: **90 %+ coverage** (opp fra ~75 %).

---

## Del 4 — Arkitektur-forbedringer

### 4.1 VectorStore-protokoll
`VectorStore` er hardkoblet til ChromaDB. Lag en `VectorStoreBackend`-protokoll slik at man kan bytte til Qdrant, Milvus, pgvector, FAISS, etc.

```python
@runtime_checkable
class VectorStoreBackend(Protocol):
    def upsert(self, ids, embeddings, documents, metadatas) -> None: ...
    def query(self, embedding, top_k, where) -> dict: ...
    def delete(self, ids) -> None: ...
    def count(self) -> int: ...
    def get_all(self) -> dict: ...
```

### 4.2 Async støtte
Alle I/O-operasjoner (embedding, vector search, HTTP) er synkrone.
For server-bruk er dette en flaskehals.
**Plan:** Legg til `async_query()`, `async_add()` basert på `asyncio`. La sync-versjoner wrappe async.

### 4.3 Streaming query-resultater
For store resultatsett (top_k=100+) bør `query()` kunne yielde resultater etterhvert.
**Plan:** `query_stream()` som returnerer `Iterator[QueryResult]`.

### 4.4 Miljøvariabel-støtte i config
Containeriserte deployments trenger env vars: `RAVENRAG_DB`, `RAVENRAG_MODEL`, etc.
**Prioritet:** env vars > CLI flags > config file > defaults.

### 4.5 Schema-validering av config
Typos som `persis_dir` i ravenrag.toml ignoreres stille.
**Fix:** Valider alle nøkler mot kjent schema, warn på ukjente.

---

## Del 5 — «Neste-nivå» features som gjør prosjektet unikt

### 5.1 🧠 Intelligent Chunking (Semantic Splitting)
I stedet for å dele på tegn/tokens, del på *semantisk endring*.
Sammenlign embedding-likhet mellom setninger — når likheten faller under en terskel, start ny chunk.
**Hvorfor unikt:** De fleste RAG-libs deler blindt. Semantisk splitting gir dramatisk bedre retrieval-kvalitet.

```python
splitter = SemanticSplitter(embedder=Embedder(), threshold=0.3)
chunks = splitter.split(doc)
```

### 5.2 📊 Retrieval Quality Metrics
Innebygd evaluering: gitt et sett med queries og forventede resultater, mål retrieval-kvalitet.

```python
from ravenrag.eval import evaluate

results = evaluate(
    index,
    queries=["What is RAG?", "How to chunk?"],
    expected_ids=[["doc1", "doc3"], ["doc7"]],
)
print(results.mrr)    # Mean Reciprocal Rank
print(results.ndcg)   # Normalized Discounted Cumulative Gain
print(results.recall)  # Recall@k
```

**Hvorfor unikt:** Ingen lett RAG-lib har innebygd eval. Brukere kan kvantifisere om config-endringer hjelper.

### 5.3 🔄 Incremental Re-indexing med Fingerprints
Track hashes av indekserte filer. Ved re-indeksering, hopp over uendrede filer.
Fjern dokumenter som ikke lenger finnes på disk.

```bash
raven index ./docs  # Første gang: indexer alt
raven index ./docs  # Andre gang: bare endrede/nye filer
```

**Hvorfor unikt:** De fleste libs re-indekserer alt fra scratch. Med 10k docs er det forskjellen på sekunder vs. minutter.

### 5.4 🪆 Parent-Child Document Retrieval
Indekser små chunks (for presisjon), men returner hele parent-dokumentet (for kontekst).

```python
index = DocumentIndex(retrieval_strategy="parent-child")
# Matcher på chunk-nivå, men returnerer hele seksjonen/dokumentet
results = index.query("auth flow", return_parent=True)
```

**Hvorfor unikt:** Løser det klassiske RAG-problemet: små chunks gir god retrieval men dårlig kontekst for LLM.

### 5.5 🌐 Multi-modal Document Support
Klargjør arkitekturen for bilder, tabeller og kode.
Start med: kode-blokker i markdown får egne chunks med `language`-metadata.

```python
# Automatisk ved .md-indeksering:
# ```python ... ``` → Document(text=code, metadata={"type": "code", "language": "python"})
```

### 5.6 📡 MCP Server (Model Context Protocol)
Eksponer RavenRAG som en [MCP](https://modelcontextprotocol.io/) server slik at Claude, Cursor, VS Code Copilot, etc. kan bruke den direkte.

```bash
raven mcp  # Starter MCP-server
```

**Hvorfor unikt:** Gjør RavenRAG til en first-class kunnskapskilde for AI-assistenter. Ingen annen lett RAG-lib har dette.

### 5.7 🔗 Knowledge Graph Overlay
Ekstraher entiteter og relasjoner fra dokumenter, bygg en lett graf.
Bruk grafen for å berike retrieval: «finn dokumenter relatert til denne entiteten».

```python
from ravenrag.knowledge import extract_entities

entities = extract_entities(docs)  # NER via spaCy eller regex
index.add_graph(entities)
results = index.query("auth", use_graph=True)  # Graph-boosted retrieval
```

### 5.8 💾 Export/Import & Backup
```bash
raven export --format jsonl > backup.jsonl
raven import backup.jsonl --collection restored
```

**Hvorfor unikt:** Portabilitet. Flytt indekser mellom maskiner, miljøer, eller backends.

---

## Del 6 — DX (Developer Experience)

### 6.1 Bedre feilmeldinger overalt
Wrap alle sentrale operasjoner i try/except med kontekst:
`"Failed to index docs/auth.md: ChromaDB collection 'main' is read-only"`

### 6.2 Structured logging
Bruk `logging` konsekvent med levels (DEBUG for intern flyt, INFO for operasjoner, WARNING for non-fatal).

### 6.3 Progress bars
Indexering av 1000 filer bør vise progress (`tqdm` eller lignende, valgfri dep).

### 6.4 `raven doctor`
En diagnostikk-kommando som sjekker:
- ChromaDB-tilkobling
- Embedding-model tilgjengelighet
- Disk-plass
- Config-validering

### 6.5 README badges
```md
![CI](https://github.com/egkristi/ravenrag/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/ravenrag)
![Python](https://img.shields.io/pypi/pyversions/ravenrag)
```

---

## Del 7 — Prosjekthygiene

| Oppgave | Fil |
|---------|-----|
| Legg til `.gitattributes` | `.gitattributes` |
| Legg til `CHANGELOG.md` | `CHANGELOG.md` |
| Legg til `CONTRIBUTING.md` | `CONTRIBUTING.md` |
| Legg til `SECURITY.md` | `SECURITY.md` |
| Legg til GitHub Issue templates | `.github/ISSUE_TEMPLATE/` |
| Legg til PR template | `.github/PULL_REQUEST_TEMPLATE.md` |
| Legg til `py.typed` marker | `ravenrag/py.typed` |
| Legg til mypy i CI | `.github/workflows/ci.yml` |

---

## Prioritert implementeringsrekkefølge

| Fase | Hva | Effort | Impact |
|------|-----|--------|--------|
| **A — Robusthet** | Timeouts, request limits, input-validering, path traversal fix | Liten | Kritisk |
| **B — Tester** | test_store, test_watcher, server HTTP-tester, 90 %+ coverage | Middels | Høy |
| **C — Incremental re-indexing** | Fingerprint-tracking, skip uendrede filer | Middels | Høy |
| **D — Semantic splitting** | Embedding-basert chunk-splitting | Middels | Høy |
| **E — VectorStore-protokoll** | Abstraher bort ChromaDB, muliggjør andre backends | Middels | Høy |
| **F — MCP server** | Model Context Protocol for AI-assistenter | Middels | Unik |
| **G — Retrieval eval** | Innebygd MRR/NDCG/Recall evaluering | Middels | Unik |
| **H — Parent-child retrieval** | Søk på chunk, returner parent | Middels | Unik |
| **I — Export/import** | JSONL backup/restore | Liten | Middels |
| **J — Async API** | asyncio-baserte operasjoner | Stor | Middels |
| **K — Knowledge graph** | Entity extraction + graph-boosted retrieval | Stor | Unik |
| **L — Prosjekthygiene** | CHANGELOG, CONTRIBUTING, badges, mypy, py.typed | Liten | Middels |

---

## Anbefaling

Start med **Fase A + B + L** (robusthet, tester, hygiene) — dette er grunnmuren.
Deretter **C + D** (incremental re-indexing + semantic splitting) — dette er det som skiller RavenRAG fra andre lette RAG-libs.
Så **F** (MCP server) — dette er killer-featuren som gjør prosjektet virkelig unikt i markedet.
