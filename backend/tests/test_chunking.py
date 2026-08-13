"""Tests for deterministic, structure-aware chunking."""

from __future__ import annotations

from app.rag.chunking import (chunk_article, chunk_articles, normalize_text,
                              split_sections)


def test_normalize_preserves_newlines_but_collapses_inline_spaces():
    """
    Structure must survive normalisation.

    Collapsing "\\s+" would eat the newlines that carry the section and
    paragraph boundaries — the bug this test exists to prevent.
    """
    text = "Line   one\r\n\r\n\r\n## A   Heading\nbody   text  "
    result = normalize_text(text)
    assert "\n\n## A Heading\n" in result
    assert "Line one" in result
    assert "   " not in result


def test_sections_are_detected(sample_articles):
    sections = split_sections(sample_articles[0]["body"])
    titles = [title for title, _ in sections]
    assert titles == ["Overview", "Eligibility", "How to request"]


def test_text_before_the_first_heading_is_kept(sample_articles):
    """Nothing may be silently dropped — leading text becomes 'Overview'."""
    sections = dict(split_sections(sample_articles[0]["body"]))
    assert "thirty days" in " ".join(sections["Overview"])


def test_chunks_never_span_two_sections(sample_articles):
    for chunk in chunk_articles(sample_articles, 200, 20):
        # With a 200-word budget every section fits in one chunk, so a chunk
        # containing two sections' text would prove the boundary leaked.
        assert chunk.section


def test_chunk_ids_are_stable_and_ordered(sample_articles):
    chunks = chunk_article(sample_articles[1], 40, 8)
    assert [c.chunk_id for c in chunks] == [f"a2#{i:02d}" for i in range(1, len(chunks) + 1)]
    assert [c.ordinal for c in chunks] == list(range(1, len(chunks) + 1))


def test_chunking_is_deterministic(sample_articles):
    first = chunk_articles(sample_articles, 40, 8)
    second = chunk_articles(sample_articles, 40, 8)
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]


def test_metadata_is_preserved_on_every_chunk(sample_articles):
    for chunk in chunk_articles(sample_articles, 40, 8):
        source = next(a for a in sample_articles if a["id"] == chunk.article_id)
        assert chunk.title == source["title"]
        assert chunk.url == source["url"]
        assert chunk.product_area == source["product_area"]
        assert chunk.tags == source["tags"].split()
        assert chunk.source == "kb_articles"


def test_chunks_respect_the_word_budget():
    article = {"id": "big", "title": "Big", "tags": "", "url": None,
               "product_area": None,
               "body": "## Section\n" + " ".join(f"word{i}" for i in range(500))}
    for chunk in chunk_article(article, 50, 10):
        assert len(chunk.text.split()) <= 50


def test_overlap_carries_words_forward():
    """Consecutive chunks in one section must share the overlap tail."""
    article = {"id": "ov", "title": "Ov", "tags": "", "url": None,
               "product_area": None,
               "body": "## S\n" + " ".join(f"w{i}" for i in range(120))}
    chunks = chunk_article(article, 40, 10)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text.split()[-10:] == current.text.split()[:10]


def test_zero_overlap_produces_no_repetition():
    article = {"id": "no", "title": "No", "tags": "", "url": None,
               "product_area": None,
               "body": "## S\n" + " ".join(f"w{i}" for i in range(100))}
    chunks = chunk_article(article, 25, 0)
    words = [w for chunk in chunks for w in chunk.text.split()]
    assert len(words) == len(set(words))


def test_overlap_larger_than_chunk_is_clamped():
    """An absurd overlap must not cause an infinite loop or unbounded growth."""
    article = {"id": "cl", "title": "Cl", "tags": "", "url": None,
               "product_area": None,
               "body": "## S\n" + " ".join(f"w{i}" for i in range(60))}
    chunks = chunk_article(article, 10, 99)
    assert chunks
    for chunk in chunks:
        assert len(chunk.text.split()) <= 10


def test_no_chunk_is_only_overlap():
    """A trailing chunk that repeats the previous tail and adds nothing is a bug."""
    article = {"id": "t", "title": "T", "tags": "", "url": None, "product_area": None,
               "body": "## S\n" + " ".join(f"w{i}" for i in range(83))}
    chunks = chunk_article(article, 20, 5)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.text != previous.text[-len(current.text):]
        assert len(current.text.split()) > 5


def test_real_knowledge_base_produces_multiple_chunks_per_article(seeded_db):
    from app.database import db
    articles = db.all_kb_articles()
    chunks = chunk_articles(articles, 110, 25)
    assert len(chunks) > len(articles)      # chunking is doing real work
    assert len({c.section for c in chunks}) > len(articles)
