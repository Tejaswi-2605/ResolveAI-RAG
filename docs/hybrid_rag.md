# Hybrid RAG — how retrieval works, and what the numbers say

This is the deep dive. [README.md](../README.md) explains the concepts from
zero; this file explains the *decisions*, and shows the measurements that drove
them — including the two places where the measurement said "no".

---

## 1. The pipeline

```
                         user query
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │  BM25 (lexical)  │            │ embed the query  │
    │  exact tokens    │            │  384 numbers     │
    └────────┬─────────┘            └────────┬─────────┘
             │                               ▼
             │                      ┌──────────────────┐
             │                      │ FAISS IndexFlatIP│
             │                      │  cosine search   │
             │                      └────────┬─────────┘
             │  top 10                       │  top 10
             └───────────────┬───────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  weighted RRF        │  merge two rankings using RANK,
                  │  Σ wᵢ /(k + rankᵢ)   │  never raw scores
                  └──────────┬───────────┘
                             ▼
                  ┌─────────────────────┐
                  │  reranker (optional) │  OFF by default — see §6
                  └──────────┬───────────┘
                             ▼
                     top-4 evidence chunks
                             │
                             ▼
                  fenced as DATA, labelled [chunk_id]
                             │
                             ▼
                            LLM
                             │
                             ▼
                     citation validation
```

Code: [`backend/app/rag/hybrid.py`](../backend/app/rag/hybrid.py) orchestrates;
each stage lives in its own module.

---

## 2. Why two retrieval arms

The two methods fail in **uncorrelated** ways. That is the entire argument for
combining them — combining two retrievers that fail on the same queries buys
nothing.

| | BM25 (lexical) | Embeddings (semantic) |
|---|---|---|
| Matches | literal tokens | meaning |
| Strong at | `ERR-4029`, `X-Signature`, SAML, invoice ids | "two people left the company" → "deactivate a seat" |
| Blind to | any paraphrase | rare exact strings |
| Needs | nothing | a 90 MB model |
| Cost | ~0.05 ms | ~10 ms |

**The failure that motivates BM25.** An embedding compresses text into 384
numbers. Rare, information-dense strings are exactly what gets compressed away,
so an embedding model happily decides `ERR-4029` and `ERR-3007` are nearly the
same thing — both are short alphanumeric error codes. Our measurements confirm
it: on the 10 exact-identifier queries, semantic-only scores **0.900** recall@1
while lexical scores **1.000**.

**The failure that motivates embeddings.** On the 12 paraphrase queries,
lexical-only scores **0.583** recall@1 against semantic's **0.917**. A customer
writing "we are still paying for people who left" shares no content words with
documentation that says "deactivate a seat".

Neither method is good enough alone. That is what "hybrid" means here, and it
is a claim the evaluation checks rather than a label.

---

## 3. Chunking

[`backend/app/rag/chunking.py`](../backend/app/rag/chunking.py)

A whole article is a bad retrieval unit: one vector averaging five topics
matches none of them well, and most of the LLM's context gets spent on text the
customer did not ask about. So articles are cut into **chunks**.

The chunker is **structure-aware**, not blind:

| boundary | behaviour |
|---|---|
| `## Section` heading | hard boundary — a chunk never spans two sections |
| blank line (paragraph) | paragraphs are packed whole when they fit |
| sentence | used only when one paragraph exceeds the budget |
| word | last resort, for a single enormous sentence |

Overlap (25 words) is carried between consecutive chunks **within the same
section**, so a sentence sitting on a boundary still appears whole somewhere.
It never crosses a section boundary — carrying refund text into a section about
API keys would poison retrieval.

Current corpus: **16 articles → 82 chunks**, averaging **38 words** each.

> **A bug worth knowing about.** An earlier version normalised whitespace with
> `re.sub(r"\s+", " ", text)`. That collapses newlines too, which destroyed
> every `## Section` heading and turned each article into one giant paragraph —
> silently, with no error. `normalize_text()` now collapses spaces *within* a
> line and preserves the line structure. `test_normalize_preserves_newlines_but_collapses_inline_spaces`
> exists to stop it coming back.

Chunking is **deterministic**: no randomness, no set iteration, no clock. The
same corpus always produces byte-identical chunks, which is what makes the
manifest fingerprint meaningful.

---

## 4. Embeddings

[`backend/app/rag/embeddings.py`](../backend/app/rag/embeddings.py)

**Model: `sentence-transformers/all-MiniLM-L6-v2`** — 384 dimensions, ~90 MB,
CPU-only, downloaded once then fully offline. Chosen because it is trained
specifically for sentence similarity, it is small enough to run on any laptop,
and it needs no API key. Configurable via `EMBEDDING_MODEL`; nothing hard-codes
it.

Every vector is **L2-normalised** (scaled to length 1). After that, the dot
product *is* the cosine similarity — which is why the vector index can use a
plain inner-product search and still be doing exact cosine ranking.

### A measured blind spot

Testing the model honestly rather than assuming it works:

| query | vs related sentence | vs unrelated sentence | verdict |
|---|---|---|---|
| "someone left the company and we still pay for them" | **0.211** | 0.005 | ✅ |
| "log in with our company account" | **0.412** | 0.077 | ✅ |
| "our bill is overdue" | **0.312** | 0.146 | ✅ |
| "stop being billed for unused licences" | **0.428** | 0.012 | ✅ |
| "we are paying for licences nobody uses" | 0.014 | **0.038** | ❌ **wrong way round** |

The last row is real and it is in the test suite as
`test_real_model_has_documented_blind_spots`. The bare noun "licences" does not
reliably connect to "seat" for this small model; add surrounding context and it
gets it right. The lesson is not "the model is bad" — it is that short,
keyword-ish queries are exactly where the **lexical arm has to carry the
result**, which is another argument for hybrid.

### The hashing provider

`HashingEmbeddings` hashes words and character n-grams into fixed buckets. It
needs no PyTorch and no download, so tests and CI stay fast and hermetic.

**It is not semantic.** It measures vocabulary overlap. It cannot tell that
"licences" and "seats" are related. It exists to exercise the plumbing
deterministically — never to claim quality. Every number in this document comes
from the real model.

### Why the factory raises instead of substituting

`get_embedding_provider()` **raises** `EmbeddingUnavailable` when
sentence-transformers is missing. It does not quietly fall back to hashing,
because that would let the system keep reporting "hybrid" while doing no
semantic retrieval at all. The retriever catches the exception, drops to
lexical, and **records the fallback in the trace**.

---

## 5. RRF — why not just add the scores?

The obvious idea is `0.5 × bm25 + 0.5 × cosine`. It does not work, because the
two numbers are not comparable:

- BM25 is **unbounded** and corpus-dependent. A score of 14 means nothing on
  its own.
- Cosine sits in **[-1, 1]**.

Any fixed weighting is a hidden bet on the score distributions, and it breaks
when the corpus changes.

**Reciprocal Rank Fusion** throws the scores away and uses only **rank** — the
one thing both arms agree on the meaning of:

```
RRF(d) = Σ  weightᵢ / (k + rank of d in list i)
         i
```

- A chunk ranked #1 by both arms scores `2/(k+1)` — the maximum.
- A chunk ranked #1 by one arm and absent from the other still scores
  `1/(k+1)`, so a strong single-arm result is not discarded.
- `k = 60` (the value from the original TREC paper) damps the difference
  between top ranks: 1/61 vs 1/62 is a small gap, so **agreement across arms
  matters more than a one-place lead within one arm**.

### Weighted RRF — chosen by measurement

Textbook RRF weights both arms 1.0, which assumes they are equally trustworthy.
On this corpus that assumption is wrong, and it costs accuracy. Measured sweep,
all 32 queries:

| lexical weight | recall@1 | recall@3 | MRR@10 | exact R@1 | paraphrase R@1 |
|---|---|---|---|---|---|
| 1.0 (textbook) | 0.875 | 1.000 | 0.927 | 1.000 | 0.667 |
| **0.8 (shipped)** | **0.938** | **1.000** | **0.958** | **1.000** | **0.833** |
| 0.6 | 0.906 | 1.000 | 0.943 | 0.900 | 0.833 |
| 0.5 | 0.906 | 1.000 | 0.948 | 0.900 | 0.833 |
| 0.4 | 0.906 | 1.000 | 0.948 | 0.900 | 0.833 |
| 0.3 | 0.906 | 1.000 | 0.948 | 0.900 | 0.833 |

**0.8 is the peak.** Below it the lexical arm loses too much influence and
exact-identifier recall@1 drops from 1.000 to 0.900 — the very thing BM25 is
there for. Configurable via `RRF_WEIGHT_LEXICAL` / `RRF_WEIGHT_SEMANTIC`.

---

## 6. The reranker — and why it ships **disabled**

[`backend/app/rag/reranker.py`](../backend/app/rag/reranker.py)

The retrieve-then-rerank pattern is standard: run something cheap over the
whole corpus, then something expensive over the survivors. A `Reranker`
protocol with three implementations (`heuristic`, `cross-encoder`, `none`) is
implemented, unit-tested and selectable by config.

**And the default is `none`, because the evaluation said so.**

With textbook RRF weights, the heuristic reranker *lowered* recall@1 from 0.938
to 0.812. After tuning the RRF weights, it changed the chunk ordering on
**0 of 32 queries** — a complete no-op.

Two reasons, both worth understanding:

1. **Its signals are bag-of-words.** A paraphrase match found by the semantic
   arm has low term coverage *by definition* — that is precisely why the
   embedding model was needed. So a lexical-flavoured reranker systematically
   penalises the results semantic retrieval exists to contribute.

2. **Min-max normalisation amplifies noise.** RRF scores are all very close
   (1/61, 1/62, 1/63). Stretching them across [0, 1] turns a one-place rank
   difference into a large score difference. It magnifies what is nearly noise.

Shipping a stage that sounds impressive and measurably makes results worse
would be the dishonest choice. It stays in the codebase because the honest fix
is a **cross-encoder** — a transformer that reads query and chunk together in
one pass and can therefore judge relevance properly — and the architecture must
make that a config change rather than a rewrite. Set `RERANKER=cross-encoder`
and re-run the evaluation.

---

## 7. Results

**Setup:** 32 labelled queries (10 exact-identifier, 12 paraphrase, 10 mixed),
16 articles → 82 chunks, all-MiniLM-L6-v2 (384d) + FAISS `IndexFlatIP`,
top-K = 5. Gold labels are at **article** level: a query counts as answered when
any chunk of a labelled article is retrieved.

### Overall

| Metric | lexical | semantic | **hybrid** | hybrid+rerank |
|---|---|---|---|---|
| recall@1 | 0.812 | 0.938 | **0.938** | 0.938 |
| recall@3 | 0.969 | 1.000 | **1.000** | 1.000 |
| recall@5 | 0.969 | 1.000 | **1.000** | 1.000 |
| MRR@10 | 0.885 | 0.969 | **0.958** | 0.958 |
| avg latency | 0.05 ms | 11.80 ms | **9.61 ms** | 9.83 ms |
| misses | 1 | 0 | **0** | 0 |

### By query kind — where the story actually is

| | lexical | semantic | **hybrid** |
|---|---|---|---|
| **exact** (10) recall@1 | **1.000** | 0.900 | **1.000** |
| **paraphrase** (12) recall@1 | 0.583 | **0.917** | 0.833 |
| **mixed** (10) recall@1 | 0.900 | **1.000** | **1.000** |

### Reading this honestly

**What hybrid wins.** It is the only configuration that is at or near the top on
*every* query kind. Lexical collapses on paraphrases (0.583). Semantic drops on
exact identifiers (0.900) — the error-code lookups that are a large share of
real support traffic. Hybrid is 1.000 on exact, 1.000 on mixed, 0.833 on
paraphrase, with **zero misses** across all 32 queries.

**What hybrid does not win.** Semantic-only edges it on overall MRR (0.969 vs
0.958) and on paraphrase recall@1 (0.917 vs 0.833). At **recall@3 and recall@5
they are identical at 1.000**, and since the agent is given the top **4** chunks,
recall@3–5 is the metric that actually governs answer quality. Within the window
that matters, hybrid and semantic are tied.

**So why ship hybrid?** Three reasons the averages do not show:

1. **Exact identifiers.** 1.000 vs 0.900 on error codes. A support system that
   fumbles one `ERR-` lookup in ten is failing at its most mechanical task.
2. **Robustness.** BM25 needs no model, no index, no download. When the vector
   index is missing or the model will not load, the lexical arm keeps answering
   — degraded and clearly labelled, but answering. Semantic-only has no such
   floor.
3. **Speed.** 9.61 ms vs 11.80 ms, because the lexical arm is essentially free
   and narrows the work.

**The honest caveat.** 32 queries over 16 articles is a small benchmark, and I
wrote both the queries and the corpus. The paraphrase queries were deliberately
written to avoid documentation vocabulary, which is *why* lexical scores 0.583
there — that number reflects a designed stress test, not a natural traffic mix.
These results show the architecture behaves as designed and the trade-offs are
real; they are not a claim about production performance.

---

## 8. Fallbacks — degradation is always visible

The rule: **retrieval degrades rather than failing, and never lies about it.**

| what breaks | what happens | recorded in the trace |
|---|---|---|
| vector index missing | lexical only | `vector_index_unavailable` |
| embedding model unavailable | lexical only | `embedding_provider_unavailable` |
| model throws mid-query | lexical only | `semantic_search_failed` |
| chunk store missing | re-chunked from SQLite, lexical only | `chunk_store_rebuilt_from_database` |
| index points at chunks that no longer exist | flagged as stale | `vector_index_stale` |
| faiss not installed | numpy backend (identical results) | logged warning |

`mode_used` is derived from **what actually produced candidates**, never from
what was requested. If you ask for hybrid and only BM25 ran, the trace says
`lexical`. That is the anti-lying property, and `test_fallback.py` (14 tests)
enforces it.

Every fallback is written to the `retrievals` table, so "how often did semantic
retrieval fall back last week?" is a SQL query, not a guess.

---

## 9. Tuning guide

| symptom | knob | direction |
|---|---|---|
| answers cut off mid-thought | `CHUNK_SIZE_WORDS` | increase |
| retrieved chunks cover several topics | `CHUNK_SIZE_WORDS` | decrease |
| answers split across two chunks | `CHUNK_OVERLAP_WORDS` | increase |
| error-code lookups miss | `RRF_WEIGHT_LEXICAL` | increase toward 1.0 |
| paraphrases miss | `RRF_WEIGHT_LEXICAL` | decrease toward 0.5 |
| right answer is present but ranked low | `TOP_K_FINAL` | increase, or enable a reranker |
| irrelevant chunks reach the LLM | `MIN_SEMANTIC_SCORE` | increase |
| retrieval too slow at scale | `TOP_K_LEXICAL` / `TOP_K_SEMANTIC` | decrease |

**Change one knob, then re-run `python evaluation/retrieval_eval.py`.** Every
default in this project was set that way, and two of them were set *against*
the intuitive choice because the numbers disagreed with it.

---

## 10. What this would need at real scale

Current design is right for tens to low thousands of chunks. Beyond that:

- **BM25 in memory → SQLite FTS5 or Elasticsearch.** Same `search()` signature;
  nothing above `lexical.py` changes.
- **`IndexFlatIP` → HNSW or IVF.** Approximate search trades a little recall for
  a lot of speed, and only starts paying off in the millions of vectors.
- **Full rebuild → incremental ingest.** Currently one changed article rebuilds
  all 82 chunks. Fine at 50 ms; not fine at a million.
- **Add a cross-encoder reranker.** The measured gap this project has not closed.
- **Chunk-level gold labels.** Article-level labels cannot distinguish "found
  the right article" from "found the right paragraph".
