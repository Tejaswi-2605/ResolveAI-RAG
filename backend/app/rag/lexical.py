"""
lexical.py — BM25 KEYWORD SEARCH, WRITTEN OUT IN FULL.

WHAT LEXICAL RETRIEVAL IS
Matching on the literal words. No neural network, no training, no GPU — just
counting. It is the half of hybrid search that embeddings are bad at.

WHY KEEP IT WHEN WE HAVE EMBEDDINGS?
Because an embedding model compresses text into ~384 numbers, and rare exact
strings are exactly what gets compressed away. Ask an embedding model whether
"ERR-4029" and "ERR-3007" are similar and it will say yes — they are both short
alphanumeric error codes. BM25 says no, because they are different tokens.
Lexical search is therefore the right tool for:

    error codes (ERR-4029)      product nouns (SAML, webhook)
    header names (X-Signature)  policy terms quoted verbatim
    identifiers (inv_003)       exact phrases the customer copied from a log

WHAT BM25 ACTUALLY COMPUTES
For a query term `t` and a document `d`, BM25 asks three questions:

  1. How RARE is the term in the corpus?  →  inverse document frequency (IDF).
     A term in 2 of 60 chunks is far more informative than one in 55 of 60.

  2. How OFTEN does it appear in this document?  →  term frequency (TF), with
     SATURATION. The tenth occurrence adds much less than the second; `k1`
     controls how fast that flattens out. (Plain TF-IDF has no saturation,
     which is one reason BM25 beats it.)

  3. Is the document LONG?  →  length normalisation. A 500-word chunk mentions
     everything more often just by being long, so its scores are damped
     relative to the average chunk. `b` controls how strongly.

    score(d, q) = Σ  IDF(t) · ( f(t,d) · (k1 + 1) )
                 t∈q         ─────────────────────────────────────────
                              f(t,d) + k1 · (1 − b + b · |d| / avgdl)

`k1 = 1.5` and `b = 0.75` are the standard defaults and are what Lucene,
Elasticsearch and rank_bm25 use.

WHY HAND-WRITTEN INSTEAD OF `pip install rank_bm25`?
Two reasons. It is sixty lines, so the dependency buys nothing. And the
TOKENISER is the part that actually decides whether "ERR-4029" is findable —
that is a domain decision about our corpus, not something to inherit from a
library's defaults.

THE TOKENISER, AND WHY IT MATTERS MOST
`ERR-4029` is emitted as THREE tokens: "err-4029", "err" and "4029". So the
document matches whether the customer writes the full code, mentions "4029"
alone, or pastes it inside a sentence. The same applies to
"x-ratelimit-remaining". Get this wrong and BM25's whole advantage disappears.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.rag.models import Chunk

# Words so common they carry no signal; keeping them would let a long chunk
# win on "the" alone. Deliberately short — over-aggressive stopword lists
# delete real query meaning.
STOPWORDS = frozenset({
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "been", "but",
    "by", "can", "do", "does", "for", "from", "get", "had", "has", "have",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not",
    "of", "on", "or", "our", "please", "so", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "up", "us", "was", "we",
    "were", "what", "when", "which", "why", "will", "with", "you", "your",
})

# A token starts with a letter or digit and may contain hyphens/underscores,
# which is what keeps "err-4029" and "x-ratelimit-remaining" in one piece.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

# Okapi BM25's standard parameters.
K1 = 1.5   # term-frequency saturation: higher = later saturation
B = 0.75   # length normalisation: 0 = off, 1 = fully proportional


def tokenize(text: str) -> list[str]:
    """
    Lower-case, drop stopwords, and split compound identifiers into sub-tokens.

    "Getting ERR-4029 from the API"
        → ["getting", "err-4029", "err", "4029", "api"]

    The compound token is kept ALONGSIDE its parts, so an exact match on
    "err-4029" still scores higher than a partial match on "4029" — the full
    token is rarer, so BM25 gives it a larger IDF.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall((text or "").lower()):
        if raw not in STOPWORDS:
            tokens.append(raw)
        if "-" in raw or "_" in raw:
            for part in re.split(r"[-_]+", raw):
                if part and part not in STOPWORDS:
                    tokens.append(part)
    return tokens


def _indexed_text(chunk: Chunk) -> str:
    """
    The text BM25 actually indexes for a chunk.

    Title, section heading and tags are prepended to the body. This is a
    cheap, transparent form of field weighting: a chunk whose TITLE contains
    the query term now contains that term one extra time, so it scores higher
    without any special-case bonus code.
    """
    return " ".join([chunk.title, chunk.section, " ".join(chunk.tags), chunk.text])


class BM25Index:
    """
    An in-memory Okapi BM25 index over a list of chunks.

    Built from the chunk store at startup and held in RAM. At this corpus size
    (tens of chunks) building takes under a millisecond, so there is no
    separate persisted artifact to keep in sync — one less thing to go stale.
    For a corpus of millions you would swap this for Elasticsearch or SQLite
    FTS5 behind the same `search()` signature; nothing above this file changes.
    """

    def __init__(self, chunks: list[Chunk], k1: float = K1, b: float = B):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        # Per-document term counts and lengths.
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        # Inverted index: term -> [(doc_index, term_frequency), ...]. Scoring a
        # query then touches only the documents that contain a query term,
        # instead of scanning the whole corpus for every term.
        self.postings: dict[str, list[tuple[int, int]]] = {}

        for position, chunk in enumerate(self.chunks):
            counts = Counter(tokenize(_indexed_text(chunk)))
            self.term_freqs.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings.setdefault(term, []).append((position, freq))

        self.doc_count = len(self.chunks)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0)

    def __len__(self) -> int:
        return self.doc_count

    def idf(self, term: str) -> float:
        """
        Inverse document frequency — how much signal this term carries.

        Uses the standard BM25+1 smoothing, `log(1 + (N - n + 0.5)/(n + 0.5))`,
        which is always positive. The unsmoothed form goes NEGATIVE for terms
        in more than half the documents, which would let a common word
        actively subtract from a document's score.
        """
        n = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.doc_count - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Rank chunks for a query.

        Returns `[(chunk_id, score), ...]`, best first, at most `top_k` long,
        and only for chunks that scored above zero. Ties break on chunk_id so
        the output is deterministic.
        """
        terms = tokenize(query)
        if not terms or not self.doc_count:
            return []

        scores: dict[int, float] = {}
        # Counting query terms means a term repeated in the query counts twice,
        # which is the behaviour a user expects when they emphasise a word.
        for term, query_freq in Counter(terms).items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for position, freq in postings:
                length_norm = 1.0 - self.b + self.b * (
                    self.doc_lengths[position] / self.avg_doc_length)
                contribution = idf * (freq * (self.k1 + 1.0)) / (
                    freq + self.k1 * length_norm)
                scores[position] = scores.get(position, 0.0) + query_freq * contribution

        ranked = sorted(
            ((self.chunks[i].chunk_id, score) for i, score in scores.items() if score > 0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:top_k]
