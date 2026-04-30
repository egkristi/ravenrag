# RavenRAG — Deep Audit & Strategic Plan v3

> Grundig dybde-analyse utført 30. april 2026 mot `main` (commit `c3fef83`, v0.5.0).
> Forrige audit (v2) dekket bug-fiks og kodekvalitet. Denne auditen fokuserer på
> arkitektur, "neste-nivå" funksjonalitet, og hva som gjør prosjektet unikt.

---

## Status quo

| Metrikk | Verdi |
|---------|-------|
| Versjon | 0.5.0 (alpha) |
| Kildekode | 2 802 linjer (17 moduler) |
| Tester | 185 (1 780 linjer, 73 % dekning) |
| Avhengigheter | 4 kjerne + 3 valgfrie |
| Python | ≥ 3.10 |
| Lisens | AGPLv3 / kommersiell dual |

### Dekningsrapport (faktisk)

| Modul | Stmts | Dekning | Merknad |
|-------|-------|---------|---------|
| `__init__.py` | 12 | 100 % | — |
| `cli.py` | 207 | 47 % | Store hull — 110 stmts mangler |
| `config.py` | 145 | 63 % | Minimal TOML-parser udekket |
| `context.py` | 31 | 94 % | OK |
| `embed.py` | 104 | 89 % | OK |
| `eval.py` | 54 | 100 % | — |
| `export.py` | 42 | 98 % | — |
| `fingerprint.py` | 46 | 100 % | — |
| `hybrid.py` | 70 | 84 % | OK |
| `index.py` | 110 | 68 % | `query_parent()` helt udekket |
| `loaders.py` | 55 | 91 % | OK |
| `mcp_server.py` | 63 | 46 % | Stdio-server udekket |
| `rerank.py` | 27 | 70 % | Minimal |
| `server.py` | 167 | 73 % | CORS/OPTIONS, edge cases |
| `splitter.py` | 140 | 92 % | OK |
| `store.py` | 49 | 96 % | — |
| `watcher.py` | 60 | 32 % | Nesten helt udekket |
| **TOTAL** | **1 382** | **73 %** | |

---

## Arkitekturanalyse

```
┌─────────────────────────────────────────────────────┐
│  Grensesnitt                                        │
│  CLI (cli.py) │ HTTP (server.py) │ MCP (mcp_server) │
├─────────────────────────────────────────────────────┤
│              DocumentIndex (index.py)                │
│  query · hybrid_query · query_parent · query_for_prompt │
├──────────┬────────────┬──────────────┬──────────────┤
│ Loaders  │ Splitters  │ Embeddings   │ Reranking    │
│ loaders  │ splitter   │ embed        │ rerank       │
├──────────┴────────────┴──────────────┴──────────────┤
│ Hybrid Search (hybrid.py) — BM25 + Vector + RRF     │
├─────────────────────────────────────────────────────┤
│ Storage: VectorStoreBackend protocol (store.py)      │
│ Backend: ChromaDB (eneste implementasjon)            │
├─────────────────────────────────────────────────────┤
│ Utilities: config · fingerprint · eval · export      │
│            context · watcher                         │
└─────────────────────────────────────────────────────┘
```

### Styrker

1. **Minimal avhengighets-graf** — 4 pakker vs LlamaIndex (~50) / LangChain (~100)
2. **Lokal-først** — ingen sky-konto nødvendig, data forlater aldri maskinen
3. **Protokoll-basert** — `EmbeddingBackend` og `VectorStoreBackend` gjør det mulig å bytte
4. **MCP-integrasjon** — fungerer med Claude, Copilot, Cursor out-of-the-box
5. **Inkrementell indeksering** — fingerprint-basert, hopper over uendrede filer
6. **Semantisk splitting** — unik chunking-strategi basert på embedding-likhet

### Svakheter

1. **Kun ChromaDB** — ingen FAISS, Qdrant, pgvector, SQLite-vec
2. **Fullstendig synkron** — ingen async/await, blokkerer I/O
3. **`query_parent()` bryter abstraksjonen** — kaller `self.store.collection.get()` direkte
4. **Ingen tråd-sikkerhet** — samtidige forespørsler kan korrumpere tilstand
5. **Ingen innebygde fil-loadere** — PDF, DOCX, HTML, Markdown nevnt men mangler
6. **Ingen query-cache** — same spørring = same embedding-beregning hver gang
7. **BM25-indeks ikke persistert** — gjenoppbygges ved endring i antall dokumenter

---

## Kritiske funn

### 🔴 Bugs og abstraksjonsbrudd

| # | Fil | Beskrivelse | Konsekvens |
|---|-----|-------------|------------|
| B1 | `index.py:200` | `query_parent()` kaller `self.store.collection.get()` — ChromaDB-spesifikk | Custom VectorStoreBackend krasjer |
| B2 | `splitter.py` | `SemanticSplitter` min-chunk-merge kan produsere chunks > `max_chunk_size` | Ugyldige chunk-størrelser |
| B3 | `config.py:75-93` | `_parse_toml_minimal()` regex-parser håndterer ikke arrays, escaped quotes, datetimes | Stille feil ved komplekse TOML-filer |

### 🟠 Sikkerhets- og robusthetsproblemer

| # | Fil | Beskrivelse |
|---|-----|-------------|
| S1 | `server.py` | Ingen rate limiting — DoS mulig via `/query` |
| S2 | `server.py` | API-nøkkel sendes over HTTP uten HTTPS-påminnelse |
| S3 | `store.py` | `where`-filter passeres direkte til ChromaDB uten validering |
| S4 | `loaders.py` | Symlink-er kan unnslippe path-traversal-beskyttelse |
| S5 | `server.py` | Ingen request timeout — treg query kan holde tråd for alltid |

### 🟡 Manglende validering

| # | Fil | Beskrivelse |
|---|-----|-------------|
| V1 | `index.py` | Ingen sjekk av embedding-vektor dimensjon ved query |
| V2 | `config.py` | Portnummer valideres ikke (0–65535) |
| V3 | `embed.py` | `encode_batched()` validerer ikke `batch_size > 0` |
| V4 | `server.py` | `/index`-endepunkt laster alt i minne — OOM ved store payloads |

---

## Konkurranseanalyse

| Egenskap | RavenRAG | LlamaIndex | LangChain | Haystack |
|----------|----------|------------|-----------|----------|
| **Lokal-først** | ✅ | ❌ Sky-først | ❌ Sky-først | ❌ Sky-først |
| **Avhengigheter** | 4 | ~50 | ~100 | ~30 |
| **Oppstartstid** | ~1s | ~5s | ~3s | ~3s |
| **Læringskurve** | Lav (3 linjer) | Middels | Høy (chains) | Høy (pipelines) |
| **Vector stores** | 1 (ChromaDB) | 20+ | 20+ | 10+ |
| **Fil-loadere** | Tekst kun | 60+ (PDF, DOCX…) | 80+ | 40+ |
| **Async** | ❌ | ✅ | ✅ | ✅ |
| **Streaming** | ❌ | ✅ | ✅ | ✅ |
| **MCP-støtte** | ✅ Innebygd | ❌ | ❌ | ❌ |
| **Evaluering** | ✅ Innebygd | Via Ragas | Via LangSmith | ❌ |
| **CLI** | ✅ Komplett | ❌ | ❌ | ❌ |

**RavenRAGs nisje**: Lettvekt, lokal-først RAG for utviklere som vil ha kontroll
uten sky-avhengigheter. Ingen andre prosjekter kombinerer MCP + CLI + eval + hybrid
search i en så liten pakke.

---

## Strategisk visjon: Hva gjør RavenRAG virkelig unikt?

De store rammeverkene (LlamaIndex, LangChain) prøver å gjøre ALT. RavenRAGs
styrke er å gjøre ÉN ting ekstremt bra: **lokal, privat, rask RAG med null
konfigurasjon**. "Neste-nivå"-funksjonaliteten handler ikke om å kopiere de
store — den handler om å utnytte lokal-først-posisjonen.

### Differensierende neste-nivå features

1. **📦 Pipeline API** — Komponer retrieval-steg som en pipeline:
   `Pipeline(Loader, Splitter, Embedder, Store).run(path)`.
   Gjør det mulig å sette opp komplekse flyter deklarativt.

2. **🔌 Pluggbar vector store** — Fiks `query_parent()`-abstraksjonen,
   legg til FAISS og SQLite-vec backends. Gjør RavenRAG brukbar uten
   ChromaDB (lettere, mer portabel).

3. **⚡ Asynkron kjerne** — `async def query()`, `async def add()`.
   Muliggjør bruk i web-servere (FastAPI, Starlette) og notebooks.
   Non-breaking: sync-wrapper for bakoverkompatibilitet.

4. **📄 Innebygde fil-loadere** — PDF (`pymupdf`), DOCX (`python-docx`),
   HTML (`beautifulsoup4`), Markdown (med frontmatter-parsing).
   Valgfrie avhengigheter, zero-config.

5. **🧠 Intelligent chunking** — Auto-detect optimal chunk-størrelse
   basert på embedding-modellens kontekstvindu. Ingen manuell tuning.

6. **💾 Query cache** — LRU-cache for embedding-beregninger og søkeresultater.
   Identiske spørringer returnerer instant (~0ms vs ~200ms).

7. **📊 Observability** — Strukturert logging med OpenTelemetry-kompatible
   spans: `index.add`, `query.embed`, `query.search`, `query.rerank`.
   Gjør det mulig å diagnostisere ytelse uten ekstern tooling.

8. **🌐 Multi-collection router** — Søk på tvers av flere collections
   med automatisk routing basert på query-innhold.

9. **🔄 Streaming results** — Generator-basert `query_stream()` som
   yielder resultater etter hvert som de rankes. Reduserer time-to-first-result.

10. **📋 Innebygd benchmarking** — `raven benchmark` CLI-kommando som
    måler indekserings-hastighet, query-latens, og minne-forbruk.
    Gjør det enkelt å sammenligne modeller og konfigurasjoner.

---

## Implementeringsplan

### Fase 1: Fundament (v0.6.0) — Fiks det som er galt

**Mål**: Fikse alle bugs, abstraksjonsbrudd og sikkerhetsproblemer.

| # | Oppgave | Fil(er) | Prioritet |
|---|---------|---------|-----------|
| 1.1 | Flytt `query_parent()` til `VectorStoreBackend`-protokollen — legg til `get_by_ids()` metode | `store.py`, `index.py` | 🔴 |
| 1.2 | Fiks `SemanticSplitter` chunk-størrelse — post-processing som håndhever `max_chunk_size` | `splitter.py` | 🔴 |
| 1.3 | Erstatt `_parse_toml_minimal()` med `tomli` fallback (for Python < 3.11) | `config.py`, `pyproject.toml` | 🟠 |
| 1.4 | Legg til port-validering (0–65535) i config | `config.py` | 🟡 |
| 1.5 | Legg til `batch_size > 0` validering i `encode_batched()` | `embed.py` | 🟡 |
| 1.6 | Legg til embedding-dimensjon sjekk ved query | `index.py` | 🟡 |
| 1.7 | Symlink-håndtering i `load_directory()` — resolv og sjekk mot rot | `loaders.py` | 🟠 |
| 1.8 | Dokumenter HTTPS-krav i server docstring og README | `server.py`, `README.md` | 🟠 |

### Fase 2: Pluggbar Storage (v0.6.0)

**Mål**: Gjøre det mulig å bruke RavenRAG uten ChromaDB.

| # | Oppgave | Fil(er) |
|---|---------|---------|
| 2.1 | Utvid `VectorStoreBackend` med `get_by_ids(ids) -> Dict` | `store.py` |
| 2.2 | Implementer `FaissStore` (in-memory, rask, ingen server) | `stores/faiss.py` (ny) |
| 2.3 | Implementer `SqliteVecStore` (SQLite-basert, null dependencies) | `stores/sqlite_vec.py` (ny) |
| 2.4 | Refaktorer `query_parent()` til å bruke `get_by_ids()` | `index.py` |
| 2.5 | Auto-velg backend basert på installerte pakker | `store.py` |

### Fase 3: Fil-loadere (v0.7.0)

**Mål**: Støtte de vanligste filformatene ut av boksen.

| # | Oppgave | Avhengighet |
|---|---------|-------------|
| 3.1 | PDF-loader med `pymupdf4llm` (markdown-output) | `pymupdf4llm` (valgfri) |
| 3.2 | DOCX-loader med `python-docx` | `python-docx` (valgfri) |
| 3.3 | HTML-loader med `beautifulsoup4` (strip tags, beholde struktur) | `beautifulsoup4` (valgfri) |
| 3.4 | Markdown-loader med frontmatter-parsing (YAML metadata → doc metadata) | innebygd |
| 3.5 | Auto-detect filtype og velg loader | `loaders.py` |
| 3.6 | Legg til `[loaders]` optional dependency group i pyproject.toml | `pyproject.toml` |

### Fase 4: Query Cache & Ytelse (v0.7.0)

**Mål**: Gjøre gjentatte søk instant og forbedre generell ytelse.

| # | Oppgave |
|---|---------|
| 4.1 | LRU-cache for embedding-beregninger (nøkkel: hash av query-tekst + modell) |
| 4.2 | Valgfri resultat-cache med TTL for `query()` og `hybrid_query()` |
| 4.3 | Persistere BM25-indeks til disk (gjenbruk mellom prosesser) |
| 4.4 | Parallell fil-indeksering i CLI (`concurrent.futures.ThreadPoolExecutor`) |
| 4.5 | Progress-bar i CLI for indeksering (via `rich` eller innebygd) |

### Fase 5: Async kjerne (v0.8.0)

**Mål**: Gjøre alle I/O-operasjoner asynkrone.

| # | Oppgave |
|---|---------|
| 5.1 | `async def aquery()`, `async def aadd()` i `DocumentIndex` |
| 5.2 | Async embedding for remote backends (Ollama, OpenAI, vLLM) |
| 5.3 | Sync-wrappers (`query()` kaller `asyncio.run(aquery())`) for bakoverkompatibilitet |
| 5.4 | Valgfri FastAPI/Starlette server som alternativ til stdlib HTTPServer |
| 5.5 | Async file watcher |

### Fase 6: Pipeline API (v0.8.0)

**Mål**: Deklarativ pipeline for komplekse flyter.

```python
from ravenrag import Pipeline, PDFLoader, SemanticSplitter, Embedder, VectorStore

pipe = Pipeline([
    PDFLoader(),
    SemanticSplitter(max_chunk_size=512),
    Embedder(model="all-MiniLM-L6-v2"),
    VectorStore(persist_dir="./db"),
])
pipe.run("./documents/")              # indekser alt
results = pipe.query("hva er RAG?")   # søk
```

| # | Oppgave |
|---|---------|
| 6.1 | `Pipeline` klasse med `run()`, `query()`, `stream()` |
| 6.2 | Steg-protokoll: `PipelineStep` med `process(docs) -> docs` |
| 6.3 | YAML-konfigurasjon av pipelines (alternativ til Python-kode) |
| 6.4 | Innebygd retry/error-handling per steg |

### Fase 7: Observability & Benchmarking (v0.9.0)

**Mål**: Gjøre det enkelt å forstå og optimalisere ytelse.

| # | Oppgave |
|---|---------|
| 7.1 | Strukturert logging med kontekst (query-id, varighet, antall resultater) |
| 7.2 | `raven benchmark` CLI — måle indeksering, query-latens, minne |
| 7.3 | Timing-dekorator for kritiske metoder (embed, search, rerank) |
| 7.4 | Valgfri OpenTelemetry-eksport for tracing |
| 7.5 | `/metrics` endpoint i HTTP-server (Prometheus-kompatibel) |

### Fase 8: Avansert Retrieval (v0.9.0)

**Mål**: Mer sofistikerte søke-strategier.

| # | Oppgave |
|---|---------|
| 8.1 | Multi-collection søk med automatisk routing |
| 8.2 | Streaming `query_stream()` — generator som yielder resultater |
| 8.3 | Multi-query expansion (omskriv spørring til flere varianter) |
| 8.4 | Kontekstuell kompresjon (fjern irrelevante deler av resultater) |
| 8.5 | Metadata-sortering post-search (dato, kilde, relevans) |

### Fase 9: Test & Kvalitet (løpende)

**Mål**: 85 %+ dekning, robusthet-tester.

| # | Oppgave | Nåværende → Mål |
|---|---------|-----------------|
| 9.1 | `cli.py` integrasjonstester | 47 % → 80 % |
| 9.2 | `mcp_server.py` tool-call tester | 46 % → 80 % |
| 9.3 | `watcher.py` med mock watchfiles | 32 % → 70 % |
| 9.4 | `index.py` query_parent, batch-add | 68 % → 90 % |
| 9.5 | `server.py` CORS, OPTIONS, edge cases | 73 % → 90 % |
| 9.6 | Concurrent access stresstest | ny |
| 9.7 | Stor-skala test (10k+ dokumenter) | ny |
| 9.8 | Property-based testing (Hypothesis) for splitters | ny |

### Fase 10: Dokumentasjon & Markedsføring (v1.0.0)

| # | Oppgave |
|---|---------|
| 10.1 | Ytelsessammenligninger vs LlamaIndex/LangChain (latens, minne) |
| 10.2 | Troubleshooting-seksjon i README |
| 10.3 | Sikkerhets- og personvern-erklæring |
| 10.4 | Interaktiv tutorial / cookbook |
| 10.5 | Publiser til PyPI |
| 10.6 | GitHub Actions: auto-publisering ved ny tag |

---

## Prioritert rekkefølge

```
v0.6.0  Fase 1 (fiks bugs) + Fase 2 (pluggbar storage) + Fase 9.1–9.5 (tester)
v0.7.0  Fase 3 (fil-loadere) + Fase 4 (cache & ytelse)
v0.8.0  Fase 5 (async) + Fase 6 (pipeline API)
v0.9.0  Fase 7 (observability) + Fase 8 (avansert retrieval)
v1.0.0  Fase 10 (docs & publisering) + Fase 9.6–9.8 (robusthet-tester)
```

---

## Oppsummering

RavenRAG har en solid kjerne og en unik posisjon: **lokal-først RAG med minimale
avhengigheter, innebygd CLI, MCP-støtte og evaluering**. Ingen andre prosjekter
kombinerer dette.

For å gå fra "lovende alpha" til "prosjektet man anbefaler" trengs:

1. **Fiks abstraksjonsbruddene** — spesielt `query_parent()` og TOML-parseren
2. **Flere vector stores** — FAISS og SQLite-vec gjør prosjektet mer portabelt
3. **Fil-loadere** — PDF og DOCX er table-stakes for RAG
4. **Query cache** — gjentatte søk bør være instant
5. **Async** — nødvendig for web-integrasjon
6. **Pipeline API** — gjør komplekse flyter deklarative

Med disse på plass har RavenRAG en reell sjanse til å bli det foretrukne
valget for utviklere som vil ha RAG uten sky-avhengigheter.
