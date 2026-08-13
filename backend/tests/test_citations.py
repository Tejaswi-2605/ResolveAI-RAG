"""
Tests for citation validation — the anti-hallucination check.

The rule under test: a citation is valid only if it names evidence ACTUALLY
RETRIEVED in this run. Not "exists in the knowledge base" — retrieved.
"""

from __future__ import annotations

from app.rag.citations import (evidence_identifiers, sources_for,
                               validate_citations)

EVIDENCE = [
    {"chunk_id": "kb_001#02", "article_id": "kb_001", "title": "Scheduling reports",
     "section": "Creating a schedule", "text": "...",
     "url": "https://docs.example.com/reports/schedule"},
    {"chunk_id": "kb_003#01", "article_id": "kb_003", "title": "API keys",
     "section": "Creating a key", "text": "...",
     "url": "https://docs.example.com/api/keys"},
]


def test_a_retrieved_chunk_id_is_valid():
    report = validate_citations(["kb_001#02"], EVIDENCE)
    assert report.valid == ["kb_001#02"]
    assert report.invalid == []
    assert report.grounded is True
    assert report.has_fabrication is False


def test_the_parent_article_id_is_also_valid():
    """Citing the article a retrieved chunk came from is coarser, not fabricated."""
    report = validate_citations(["kb_003"], EVIDENCE)
    assert report.valid == ["kb_003"]
    assert report.grounded is True


def test_a_url_from_the_evidence_is_valid():
    report = validate_citations(["https://docs.example.com/api/keys"], EVIDENCE)
    assert report.grounded is True


def test_a_fabricated_chunk_id_is_rejected():
    """The headline case: a plausible-looking id that was never retrieved."""
    report = validate_citations(["kb_999#07"], EVIDENCE)
    assert report.invalid == ["kb_999#07"]
    assert report.valid == []
    assert report.grounded is False
    assert report.has_fabrication is True


def test_a_real_article_that_was_not_retrieved_is_still_rejected():
    """
    The distinction that matters. kb_007 may well exist in the knowledge base,
    but it was not retrieved for THIS run, so citing it is fabricating a
    provenance chain.
    """
    report = validate_citations(["kb_007#01"], EVIDENCE)
    assert report.has_fabrication is True


def test_a_mix_of_valid_and_fabricated_is_split():
    report = validate_citations(["kb_001#02", "kb_999#07"], EVIDENCE)
    assert report.valid == ["kb_001#02"]
    assert report.invalid == ["kb_999#07"]
    assert report.grounded is False           # any fabrication ungrounds the run
    assert report.coverage == 0.5


def test_no_citations_means_ungrounded_but_not_fabricated():
    """An honest "I don't know" is different from an invented source."""
    report = validate_citations([], EVIDENCE)
    assert report.grounded is False
    assert report.has_fabrication is False
    assert report.coverage == 0.0


def test_every_citation_is_invalid_when_nothing_was_retrieved():
    report = validate_citations(["kb_001#02"], [])
    assert report.has_fabrication is True


def test_duplicate_citations_are_collapsed():
    report = validate_citations(["kb_001#02", "kb_001#02"], EVIDENCE)
    assert report.valid == ["kb_001#02"]


def test_object_form_citations_are_accepted():
    """Models emit either a bare string or an object; both must work."""
    report = validate_citations([{"chunk_id": "kb_001#02"}], EVIDENCE)
    assert report.valid == ["kb_001#02"]


def test_whitespace_is_tolerated():
    assert validate_citations(["  kb_001#02  "], EVIDENCE).grounded is True


def test_empty_and_null_entries_are_ignored():
    report = validate_citations(["", None, "kb_001#02"], EVIDENCE)
    assert report.valid == ["kb_001#02"]
    assert report.invalid == []


def test_identifiers_include_chunk_article_and_url():
    identifiers = evidence_identifiers(EVIDENCE)
    assert {"kb_001#02", "kb_001", "kb_003#01", "kb_003"} <= identifiers


def test_sources_are_built_from_the_evidence_only():
    """A source card can never describe an article the run did not retrieve."""
    sources = sources_for(["kb_001#02"], EVIDENCE)
    assert sources == [{
        "article_id": "kb_001", "title": "Scheduling reports",
        "chunk_id": "kb_001#02", "section": "Creating a schedule",
        "url": "https://docs.example.com/reports/schedule",
    }]


def test_sources_ignore_unknown_citations():
    assert sources_for(["kb_999#07"], EVIDENCE) == []


def test_sources_are_deduplicated():
    """Citing a chunk and its article must not produce two identical cards."""
    assert len(sources_for(["kb_001#02", "kb_001"], EVIDENCE)) == 1


def test_report_serialises_for_the_api():
    report = validate_citations(["kb_001#02", "kb_999#07"], EVIDENCE)
    assert report.to_dict() == {
        "valid": ["kb_001#02"], "invalid": ["kb_999#07"],
        "grounded": False, "coverage": 0.5, "has_fabrication": True,
    }
