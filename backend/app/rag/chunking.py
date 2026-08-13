"""
chunking.py — TURNING ARTICLES INTO RETRIEVABLE PIECES.

WHY CHUNK AT ALL?
A knowledge-base article can be a thousand words covering five different
topics. If we embed the whole article as one vector, that vector is an average
of five topics and matches none of them well. And if we hand the whole article
to the LLM, most of the context window is text the customer did not ask about.
So we cut each article into small, self-contained pieces — CHUNKS — and
retrieve chunks rather than documents.

THE TRADE-OFF, stated plainly:
  * chunks too LARGE  → diluted embeddings, wasted context, vague citations
  * chunks too SMALL  → the answer gets split across pieces and loses meaning

THIS CHUNKER IS STRUCTURE-AWARE, NOT BLIND.
A naive chunker slices every N words and happily cuts a sentence in half. This
one respects the shape the author wrote:

    "## Section" heading   → a hard boundary; chunks never span two sections
    blank line             → a paragraph boundary; paragraphs are packed whole
    sentence               → used only when one paragraph exceeds the budget
    word                   → the last resort, for a single monstrous sentence

OVERLAP: consecutive chunks in the SAME section share the last N words of the
previous chunk. That way a sentence sitting on a boundary still appears whole
somewhere. Overlap never crosses a section boundary — carrying text about
refunds into a section about API keys would poison retrieval.

DETERMINISM is a hard requirement: the same article and the same settings must
always produce byte-identical chunks. Nothing here uses randomness, hashing
order, set iteration, or wall-clock time. That is what makes the corpus
fingerprint in the manifest meaningful and the tests repeatable.
"""

from __future__ import annotations

import re

from app.rag.models import Chunk

# A run of spaces/tabs *within* a line. Newlines are handled separately,
# because the line structure is exactly what carries the section/paragraph
# information we want to preserve.
_INLINE_SPACE_RE = re.compile(r"[ \t ]+")

# Three or more newlines collapse to a paragraph break (two).
_BLANK_RUN_RE = re.compile(r"\n{3,}")

# Sentence boundary: ". ", "! " or "? " followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_SECTION_PREFIX = "## "
_DEFAULT_SECTION = "Overview"


def normalize_text(text: str) -> str:
    """
    Tidy an article body WITHOUT destroying its structure.

    Collapses runs of spaces inside a line and trims each line, but keeps
    newlines intact — the previous implementation of this project collapsed
    every whitespace character including "\\n", which silently destroyed the
    section headings and made the whole document one giant paragraph.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [_INLINE_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_RUN_RE.sub("\n\n", "\n".join(lines)).strip()


def split_sections(body: str) -> list[tuple[str, list[str]]]:
    """
    Split a normalised body into `(section_title, [paragraph, ...])` pairs.

    Text appearing before the first "## " heading belongs to a synthetic
    "Overview" section, so no content is ever dropped.
    """
    sections: list[tuple[str, list[str]]] = []
    section = _DEFAULT_SECTION
    paragraphs: list[str] = []
    buffer: list[str] = []

    def close_paragraph() -> None:
        if buffer:
            paragraphs.append(" ".join(buffer))
            buffer.clear()

    def close_section() -> None:
        close_paragraph()
        if paragraphs:
            sections.append((section, list(paragraphs)))
        paragraphs.clear()

    for line in normalize_text(body).split("\n"):
        if line.startswith(_SECTION_PREFIX):
            close_section()
            section = line[len(_SECTION_PREFIX):].strip() or _DEFAULT_SECTION
        elif not line:
            close_paragraph()
        else:
            buffer.append(line)

    close_section()
    return sections


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def _hard_split(text: str, max_words: int) -> list[str]:
    """Last resort: cut on word boundaries when one sentence exceeds the budget."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _units(paragraphs: list[str], max_words: int, overlap_words: int) -> list[str]:
    """
    Break the paragraphs of one section into packing units, each within budget.

    A paragraph stays whole if it fits. Only an oversized paragraph is split
    into sentences, and only an oversized sentence is split on words. So the
    common case preserves the author's paragraphs exactly.

    Hard splits are sized `max_words - overlap_words`, not `max_words`. A
    full-width piece would leave no room to carry the overlap tail forward, so
    a wall of unpunctuated text — the one case that reaches this path — would
    silently lose its overlap. Reserving the space keeps both guarantees: every
    chunk is within budget AND consecutive chunks share their tail.
    """
    hard_width = max(1, max_words - overlap_words)

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph.split()) <= max_words:
            units.append(paragraph)
            continue
        for sentence in _split_sentences(paragraph):
            if len(sentence.split()) <= max_words:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, hard_width))
    return units


def _pack(units: list[str], max_words: int, overlap_words: int) -> list[str]:
    """
    Greedily fill chunks up to `max_words`, carrying an overlap tail forward.

    Every unit is already within budget (see `_units`), so a flush is always
    followed by real new content in the same iteration. That is what
    guarantees we never emit a chunk consisting only of the overlap tail.
    """
    chunks: list[str] = []
    current: list[str] = []          # the words accumulated for the open chunk

    for unit in units:
        words = unit.split()
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            tail = current[-overlap_words:] if overlap_words > 0 else []
            # The budget is a hard limit, so the overlap tail is dropped when
            # carrying it would push the next chunk over. This happens with a
            # unit that already fills a whole chunk on its own — overlap is a
            # quality nicety, the word budget is a contract.
            current = tail if len(tail) + len(words) <= max_words else []
        current.extend(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_article(article: dict, chunk_size_words: int,
                  chunk_overlap_words: int) -> list[Chunk]:
    """
    Split one `kb_articles` row into Chunks, preserving every piece of metadata.

    `article` is the plain dict returned by `database.db.all_kb_articles()`.
    Chunk ids are `"<article_id>#<ordinal>"` with a zero-padded two-digit
    ordinal, so they sort correctly and read clearly in a citation.
    """
    max_words = max(1, int(chunk_size_words))
    # Overlap must be strictly smaller than the chunk, or packing never advances.
    overlap = max(0, min(int(chunk_overlap_words), max_words - 1))

    tags = [t for t in re.split(r"[\s,]+", article.get("tags") or "") if t]
    chunks: list[Chunk] = []
    ordinal = 1

    for section, paragraphs in split_sections(article["body"]):
        for text in _pack(_units(paragraphs, max_words, overlap), max_words, overlap):
            chunks.append(Chunk(
                chunk_id=f"{article['id']}#{ordinal:02d}",
                article_id=article["id"],
                title=article["title"],
                section=section,
                text=text,
                tags=tags,
                url=article.get("url"),
                product_area=article.get("product_area"),
                ordinal=ordinal,
                source="kb_articles",
            ))
            ordinal += 1

    return chunks


def chunk_articles(articles: list[dict], chunk_size_words: int,
                   chunk_overlap_words: int) -> list[Chunk]:
    """Chunk every article, preserving the order the caller supplied."""
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(chunk_article(article, chunk_size_words, chunk_overlap_words))
    return chunks
