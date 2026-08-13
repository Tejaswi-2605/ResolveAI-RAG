"""
citations.py — PROVING THE ANSWER CAME FROM THE EVIDENCE.

THE PROBLEM
Retrieval reduces hallucination; it does not eliminate it. A model handed four
chunks can still answer from its pre-training, blend two chunks into a claim
neither makes, or invent a plausible-looking source id. "kb_009#03" reads
exactly like a real citation whether or not such a chunk exists.

THE RULE THIS FILE ENFORCES
    A citation is valid only if it names evidence that was ACTUALLY RETRIEVED
    during this run.

Not "exists in the knowledge base" — RETRIEVED, in this run. That distinction
matters: a model that cites a real article it was never shown is still
fabricating a provenance chain, and this is exactly the failure a citation
check exists to catch.

WHAT HAPPENS TO AN INVALID CITATION
The agent strips it, marks the run ungrounded, and escalates to a human rather
than sending a confident answer with a fabricated source. Deleting the bad
citation alone would be worse than useless: the claim it was propping up would
still go out, just without the audit trail that reveals it is unsupported.

THE EVIDENCE LEDGER
The agent accumulates every chunk retrieved across every knowledge-tool call in
one run and validates against that union. So a citation from step 1 is still
valid at step 4 — the model is not punished for remembering.

These are PURE FUNCTIONS over plain dicts. No database, no config, no I/O — so
they are trivially testable and the agent can import them without dragging in
faiss or an embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CitationReport:
    """The verdict on one run's citations."""

    valid: list[str] = field(default_factory=list)    # cited AND retrieved
    invalid: list[str] = field(default_factory=list)  # cited but never retrieved
    grounded: bool = False       # at least one valid citation, and none invalid
    coverage: float = 0.0        # share of the cited ids that check out

    @property
    def has_fabrication(self) -> bool:
        """True when the model cited something it was never shown."""
        return bool(self.invalid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": list(self.valid),
            "invalid": list(self.invalid),
            "grounded": self.grounded,
            "coverage": round(self.coverage, 3),
            "has_fabrication": self.has_fabrication,
        }


def evidence_identifiers(evidence: list[dict]) -> set[str]:
    """
    Every identifier that legitimately refers to a piece of retrieved evidence.

    Both the chunk id ("kb_003#02") and its article id ("kb_003") count, and
    the article's URL too. A model that cites the article a retrieved chunk
    came from is being accurate at a coarser granularity, not fabricating —
    so accepting all three is correct, not lenient.
    """
    identifiers: set[str] = set()
    for item in evidence or []:
        for key in ("chunk_id", "article_id", "url"):
            value = item.get(key)
            if value:
                identifiers.add(str(value).strip())
    return identifiers


def validate_citations(cited: list, evidence: list[dict]) -> CitationReport:
    """
    Check a model's citation list against the evidence ledger.

    `cited` is whatever came back in the result's `citations` field — the model
    may emit strings or objects, so both are handled. Duplicates are collapsed
    and order is preserved, which keeps the report readable and deterministic.
    """
    allowed = evidence_identifiers(evidence)

    seen: set[str] = set()
    normalised: list[str] = []
    for entry in cited or []:
        if entry is None:
            continue          # `str(None)` would become the literal "None"
        if isinstance(entry, dict):
            value = entry.get("chunk_id") or entry.get("article_id") or entry.get("id") or ""
        else:
            value = entry
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            normalised.append(value)

    valid = [c for c in normalised if c in allowed]
    invalid = [c for c in normalised if c not in allowed]
    coverage = len(valid) / len(normalised) if normalised else 0.0

    return CitationReport(
        valid=valid,
        invalid=invalid,
        grounded=bool(valid) and not invalid,
        coverage=coverage,
    )


def sources_for(cited: list[str], evidence: list[dict]) -> list[dict]:
    """
    Expand validated citations into the source objects the UI renders.

    Returns one entry per cited chunk:
        {"article_id": "kb_001", "title": "Refund policy",
         "chunk_id": "kb_001#02", "section": "...", "url": "..."}

    Built strictly FROM the evidence, so a source card can never describe an
    article the run did not retrieve.
    """
    by_chunk = {item["chunk_id"]: item for item in evidence or [] if item.get("chunk_id")}
    by_article: dict[str, dict] = {}
    for item in evidence or []:
        by_article.setdefault(str(item.get("article_id")), item)

    sources: list[dict] = []
    seen: set[str] = set()
    for citation in cited or []:
        item = by_chunk.get(citation) or by_article.get(citation)
        if item is None:
            continue
        key = item["chunk_id"]
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "article_id": item.get("article_id"),
            "title": item.get("title"),
            "chunk_id": item.get("chunk_id"),
            "section": item.get("section"),
            "url": item.get("url"),
        })
    return sources
