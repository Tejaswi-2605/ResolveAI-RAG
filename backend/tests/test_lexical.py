"""Tests for BM25 lexical retrieval and its tokeniser."""

from __future__ import annotations

from app.rag.lexical import BM25Index, tokenize


# ── the tokeniser: where BM25's advantage is won or lost ──────────
def test_identifiers_survive_tokenisation():
    tokens = tokenize("Getting ERR-4029 from the API")
    assert "err-4029" in tokens          # the whole, rare identifier
    assert "err" in tokens               # and its parts, so partial mentions match
    assert "4029" in tokens


def test_header_names_are_split_into_parts():
    tokens = tokenize("check X-RateLimit-Remaining")
    assert "x-ratelimit-remaining" in tokens
    assert {"ratelimit", "remaining"} <= set(tokens)


def test_stopwords_are_dropped():
    assert "the" not in tokenize("the invoice and the refund")
    assert "invoice" in tokenize("the invoice and the refund")


def test_tokenise_is_case_insensitive():
    assert tokenize("ERR-4029") == tokenize("err-4029")


def test_empty_input_gives_no_tokens():
    assert tokenize("") == []
    assert tokenize(None) == []


# ── ranking ───────────────────────────────────────────────────────
def test_exact_error_code_retrieves_the_right_article(sample_chunks):
    index = BM25Index(sample_chunks)
    results = index.search("ERR-9001", top_k=3)
    assert results
    top_chunk = next(c for c in sample_chunks if c.chunk_id == results[0][0])
    assert top_chunk.article_id == "a2"
    assert "ERR-9001" in top_chunk.text


def test_scores_are_positive_and_descending(sample_chunks):
    results = BM25Index(sample_chunks).search("refund invoice eligibility", top_k=5)
    scores = [score for _, score in results]
    assert all(score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_respected(sample_chunks):
    assert len(BM25Index(sample_chunks).search("refund", top_k=1)) == 1


def test_a_query_with_no_matching_terms_returns_nothing(sample_chunks):
    assert BM25Index(sample_chunks).search("zzzz nonexistent quixotic", top_k=5) == []


def test_a_query_of_only_stopwords_returns_nothing(sample_chunks):
    assert BM25Index(sample_chunks).search("the and of to", top_k=5) == []


def test_title_and_tag_terms_are_searchable(sample_chunks):
    """Title/section/tags are indexed alongside the body — cheap field weighting."""
    results = BM25Index(sample_chunks).search("Managing team members", top_k=3)
    top = next(c for c in sample_chunks if c.chunk_id == results[0][0])
    assert top.article_id == "a3"


def test_rare_terms_outrank_common_ones(sample_chunks):
    """
    IDF at work: a term appearing in one chunk is far more informative than one
    appearing everywhere, and must dominate the ranking.
    """
    index = BM25Index(sample_chunks)
    rare = index.search("ERR-9002", top_k=1)[0]
    top_chunk = next(c for c in sample_chunks if c.chunk_id == rare[0])
    assert "ERR-9002" in top_chunk.text


def test_idf_is_never_negative(sample_chunks):
    """
    Unsmoothed IDF goes negative for terms in over half the corpus, which would
    let a common word subtract from a document's score. The smoothed form must
    not do that.
    """
    index = BM25Index(sample_chunks)
    for term in index.postings:
        assert index.idf(term) >= 0


def test_ranking_is_deterministic(sample_chunks):
    index = BM25Index(sample_chunks)
    assert index.search("refund policy", 5) == index.search("refund policy", 5)


def test_ties_break_on_chunk_id(sample_chunks):
    """Two chunks with identical text must still produce a stable order."""
    duplicated = list(sample_chunks)
    index = BM25Index(duplicated)
    first = [cid for cid, _ in index.search("refund", 5)]
    second = [cid for cid, _ in index.search("refund", 5)]
    assert first == second


def test_an_empty_corpus_does_not_crash():
    assert BM25Index([]).search("anything", 5) == []


def test_longer_documents_are_not_automatically_favoured(sample_chunks):
    """
    Length normalisation: padding a chunk with unrelated filler must not raise
    its score for a query it does not answer better.
    """
    from dataclasses import replace
    padded = [replace(chunk, text=chunk.text + " filler" * 200)
              if chunk.article_id == "a1" else chunk
              for chunk in sample_chunks]
    results = BM25Index(padded).search("ERR-9001", top_k=1)
    top = next(c for c in padded if c.chunk_id == results[0][0])
    assert top.article_id == "a2"
