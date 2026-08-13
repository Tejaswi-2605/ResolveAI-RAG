# Changelog

## 1.0.0 — ResolveAI-RAG

A clean, new project. It reuses the *architectural ideas* of the original
[ResolveAI](https://github.com/Tejaswi-2605/ResolveAi) and rewrites the code
around genuine Hybrid RAG. It is not the old repository with a vector database
bolted on, and the original is unmodified.

---

### Ideas kept from ResolveAI (rewritten, not copied)

| Idea | Why it survived |
|---|---|
| **Bounded agent loop** | Step budget, transient-only retries, argument validation, code-level auth gate. The contract "never raises for an expected failure; every path ends in a persisted, schema-valid run" is what gives the UI one shape. Extended with a fifth bound: citation validation. |
| **Tool objects + registry + `call_tool()`** | One entry point means argument validation can never be skipped. Hand-written validator, no `jsonschema` — a trust boundary should be readable. |
| **`privileged=True` on `issue_refund`** | The single most important safety property: the tool that could move money contains no database write. Kept exactly. |
| **Three-layer injection defence** | Structural fencing (text cannot close its own fence) + regex detection tuned for precision + the code gate as the real guarantee. |
| **Structured-output contract** | `extract_json` → collect *all* errors → one repair prompt → a fallback that is itself schema-valid. |
| **Versioned prompts stamped on every run** | v1 vs v2 is what makes "we can attribute behaviour to a prompt version" true rather than aspirational. |
| **Provider interface + deterministic mock** | Free, offline, reproducible tests; one place for timeouts and token accounting; vendor swap without touching the agent. |
| **stdlib `sqlite3`, parameterised SQL, `CHECK` constraints** | Nine small tables with explicit queries; an ORM would hide what a reviewer most wants to read. |
| **Approval endpoint as the only money path** | Idempotent, 409 on replay. The authorisation boundary sits at the human decision. |
| **Deterministic graders + a CI gate** | "My CI fails the build if a prompt change makes the agent less safe." |
| **Three-pane React console** | Inbox / ticket / trace rail. A support agent will not trust a draft they cannot audit. |

---

### Replaced

| Original | Now | Why |
|---|---|---|
| `search_knowledge_base` — term-frequency counting with a 3× title bonus | Full hybrid pipeline: BM25 + embeddings → weighted RRF → optional rerank | The old scorer could not find a paraphrase (no shared words) or discriminate between six articles mentioning "CSV". **One** implementation, not both. |
| 10 two-sentence KB articles | 16 sectioned articles, 250–400 words each | Every old article fitted in one chunk, so chunking was a no-op and retrieval evaluation could not distinguish a good ranker from a lucky one. The new corpus has exact identifiers where lexical wins, paraphrasable concepts where semantic wins, and overlapping vocabulary so the ranker must discriminate. |
| Citations = whatever ids the model emitted | Citations validated against an evidence ledger | A citation now means "this was actually retrieved in this run", and a fabricated one escalates the ticket. |
| `app/core/providers/` | `app/providers/` | A vendor adapter is infrastructure, not agent logic. |
| Flat `db.py` + `schema.sql` + `seed.py` | `app/database/` | One package owns persistence. |
| `backend/eval/` | `evaluation/` at the root | It evaluates the whole system, not just the backend. |

---

### New

**Hybrid RAG engine** (`backend/app/rag/`)
- `chunking.py` — deterministic, structure-aware: `##` headings are hard
  boundaries, paragraphs packed whole, overlap carried within a section only
- `embeddings.py` — `EmbeddingProvider` protocol; local sentence-transformers,
  plus a dependency-free hashing provider for tests (documented as *not*
  semantic)
- `lexical.py` — hand-written Okapi BM25 with an inverted index and a tokeniser
  that keeps `ERR-4029` findable as three tokens
- `vector_store.py` — FAISS `IndexFlatIP` and an identical numpy fallback
- `hybrid.py` — weighted RRF, the fallback ladder, and the retrieval trace
- `reranker.py` — three pluggable implementations
- `citations.py` — pure functions; the anti-hallucination check
- `ingest.py` — CLI with a corpus fingerprint for staleness detection

**Security additions**
- `wrap_evidence()` — retrieved chunks fenced as DATA with `[chunk_id]` labels
- `scan_evidence()` — detects **indirect** injection: instructions planted in a
  knowledge-base document

**Observability**
- New `retrievals` table: mode requested vs used, candidates per arm, fusion
  method, reranker, fallbacks, winning chunk ids, latency
- `agent_runs` gained `rag_mode` and `citations_valid`
- `GET /api/rag/status` — is the index present, current, and is semantic
  retrieval genuinely available?
- Agent runs return a `stages` list
- Frontend `RetrievalPanel` and `SourcesCard`

**Evaluation**
- `retrieval_eval.py` — 32 labelled queries, four configurations, broken down by
  query kind
- `run_eval.py` — 20 agent cases; compares prompt versions *and* RAG modes
- `gate.py` — 14 rules, now covering citation correctness and retrieval integrity

**Configuration** — every knob declared once in `config.py`; no magic numbers.

---

### Measured decisions

Two defaults were set **against** the intuitive choice because the evaluation
disagreed with it:

1. **RRF lexical weight 0.8, not the textbook 1.0.** Equal weighting scored
   0.875 recall@1; 0.8 scored 0.938 while keeping exact-identifier recall at
   1.000.
2. **The reranker ships disabled.** With equal RRF weights it *lowered* recall@1
   to 0.812; after weight tuning it changed the ranking on 0 of 32 queries. Its
   bag-of-words signals systematically penalise the paraphrase matches the
   semantic arm exists to find.

---

### Bugs found and fixed during the rewrite

Recorded because each one is now covered by a test:

- **Chunking destroyed document structure.** Normalising with `re.sub(r"\s+", " ")`
  collapsed newlines, erasing every `##` heading — silently, with no error.
- **BM25 scored a `Counter` object as a number**, so ranking was meaningless.
- **RRF ranks were always 1**, which made fusion a no-op.
- **`RetrievalTrace` was a frozen dataclass being mutated** — would raise at
  runtime on the first retrieval.
- **`NumpyVectorIndex.load()` dropped its chunk ids**, so a reloaded index
  mapped vectors to nothing.
- **`citations.py` had a syntax error** (`data@dataclass`).
- **`SentenceTransformerEmbeddings.embed_text` normalised on `axis=1`** for a
  1-D array.
- **The embedding fallback could never trigger** — the factory caught
  `ImportError` at construction, but construction imported nothing.
- **Overlap could push a chunk over the word budget**, and the first fix removed
  overlap entirely for unpunctuated text. Resolved by sizing hard splits to
  `max_words - overlap`.
- **`None` in a citation list became the literal string `"None"`.**
- **`extract_json` accepted a top-level array** by finding the first `{` inside it.

---

### Deliberately not carried over

- `docs/explanations/step1..step13*.md` — build-diary scaffolding, not architecture
- `TASKS.md` — a checklist for building the old project
- `eval/results/*.json` and `report.md` — stale numbers from the old system;
  regenerated, never copied
- `frontend/dist/` and `.vscode/` — build output and editor config
- The old `search_knowledge_base` scorer — replaced, not kept alongside

---

### Verified at release

- 345 backend tests passing (16 files; 3 use the real embedding model)
- 18 frontend tests passing; TypeScript clean; production build succeeds
- Retrieval evaluation: hybrid 0.938 recall@1, 1.000 recall@3, zero misses
- Agent evaluation: prompt v2 at 1.000 injection defence, 1.000 grounded
  citations, 0.000 unsupported claims
- Regression gate: 14 / 14 rules passing
