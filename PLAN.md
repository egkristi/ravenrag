# RavenRAG — Audit & Plan v2

> Dybde-analyse utført 30. april 2026 mot `main` (commit `13bf800`, v0.5.0).

---

## Status quo

| Metrikk | Verdi |
|---------|-------|
| Versjon | 0.5.0 |
| Python | ≥ 3.10 |
| Tester | 173 (168 unit + 5 integration, alle grønn) |
| Coverage | 66 % (mål: 85 %+) |
| Moduler | 17 source-filer, 16 test-filer |
| Backends | sentence-transformers, Ollama, OpenAI-compatible, vLLM |
| Ekstra | CLI, HTTP-server, MCP-server, hybrid search, reranking, semantic splitting, eval, export/import, fingerprints, watcher |

Solide nyvinninger i v0.5.0. Neste steg er å tette bugs, heve test-dekningsgrad, og polere DX.

---

## Del 1 — Bugs & kritiske funn

### 1.1 CLI fingerprint-sletting fjerner ikke dokumenter fra indeks
`raven index` detekterer slettede filer og kaller `fp_store.remove(key)`, men dokumentene ligger fortsatt i ChromaDB. Brukeren tror de er fjernet.
**Fix:** Etter `fp_store.remove(key)`, slett alle dokumenter med matchende `source` metadata fra indeksen.

### 1.2 `query_parent()` svelger alle exceptions
`except Exception:` i `query_parent()` betyr at enhver feil (f.eks. ChromaDB nede) stille faller tilbake til chunken — ingen logging, ingen advarsel.
**Fix:** Logg `logger.warning()` i except-blokken.

### 1.3 MCP-server hardkoder versjon
`"version": "0.5.0"` i `mcp_server.py` vil drifte ved neste release.
**Fix:** Bruk `from . import __version__`.

### 1.4 `get_doc_ids()` er en placeholder
`FingerprintStore.get_doc_ids()` returnerer alltid `None`. Metoden er udokumentert og aldri kalt.
**Fix:** Fjern den — den skaper forvirring.

### 1.5 Server eksponerer interne exceptions til klient
`except Exception as e: self._send_json({"error": str(e)}, 500)` i server.py kan lekke interne detaljer (stier, embedding API-nøkler, stack traces).
**Fix:** Returner generisk feilmelding, logg detaljer internt.

### 1.6 Hybrid search er ineffektiv ved lav alpha
Henter `top_k * 4` vektorresultater selv når `alpha=0.0` (ren BM25) — bortkastet arbeid.
**Fix:** Skaler vektor-henting basert på alpha.

### 1.7 BM25-indeks bygges for hvert kall
`HybridSearcher` instantierer `BM25Okapi` per query, inkludert tokenisering av alle dokumenter.
**Fix:** Bygg BM25-indeks i `__init__` og invalider ved endringer.

---

## Del 2 — Test-gap (coverage 66 % → 85 %+)

| Modul | Nå | Mål | Hva mangler |
|-------|-----|-----|-------------|
| `server.py` | 18 % | 80 % | Ekte HTTP-request tester (health, query, prompt, index, auth, CORS, 413, 404) |
| `mcp_server.py` | 45 % | 80 % | `run_stdio_server()` JSON-RPC loop (mock stdin/stdout) |
| `cli.py` | 49 % | 75 % | `index` happy-path, `export`/`import` roundtrip, `doctor` |
| `watcher.py` | 32 % | 65 % | Watch-loop med mock watchfiles (add/modify/delete) |
| `config.py` | 63 % | 85 % | TOML fallback-parser, env override edge cases |
| `index.py` | 66 % | 85 % | `query_parent()`, `query_for_prompt()`, rerank path |

---

## Del 3 — Kode-kvalitet

### 3.1 Fjern bare `except Exception:` blokker
- `index.py` `query_parent()` — bruk spesifikke exceptions + logging
- `watcher.py` — bruk spesifikke exceptions

### 3.2 Fjern ubrukt `py.typed` hvis tom
`ravenrag/py.typed` er opprettet men tom — det er korrekt per PEP 561.

### 3.3 Auto-utled `_KNOWN_KEYS` fra dataclasser
`config.py` har hardkodede `_KNOWN_KEYS`. Disse bør utledes automatisk fra `IndexConfig.__dataclass_fields__` etc.

### 3.4 Generisk server-feilmelding
Erstatt `str(e)` i server 500-responser med `"Internal server error"`.

---

## Del 4 — Implementeringsrekkefølge

| Fase | Hva | Effort |
|------|-----|--------|
| **A** | Fiks CLI fingerprint-sletting (1.1) | Liten |
| **B** | Fiks `query_parent()` logging (1.2) | Liten |
| **C** | MCP versjon fra `__version__` (1.3) | Liten |
| **D** | Fjern `get_doc_ids()` placeholder (1.4) | Liten |
| **E** | Generisk server-feilmelding (1.5) | Liten |
| **F** | Optimaliser hybrid search alpha (1.6) | Liten |
| **G** | Cache BM25-indeks (1.7) | Middels |
| **H** | Auto-utled config known_keys (3.3) | Liten |
| **I** | Hev test-coverage → 85 %+ (Del 2) | Middels |
| **J** | Lint + test + commit + push | Liten |
