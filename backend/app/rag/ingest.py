"""
ingest.py — BUILDING THE SEARCH INDEX FROM THE KNOWLEDGE BASE.

    SQLite kb_articles          (the source of truth)
            │
            ▼  normalise + chunk
        chunks.json
            │
            ▼  embed
        vectors.npy
            │
            ▼  index
    index.faiss + ids.json + manifest.json

THE PRINCIPLE THAT SHAPES THIS FILE
`kb_articles` is authoritative. Everything under `data/index/` is DERIVED — it
can be deleted at any moment and rebuilt by running this command. That is why
the index directory is git-ignored: committing it would mean committing a
cache, which then drifts out of date and starts lying.

Run it:
    python -m app.rag.ingest              # build (or rebuild) the index
    python -m app.rag.ingest --dry-run    # chunk and report, write nothing
    python -m app.rag.ingest --stats      # inspect the index that exists

THE CORPUS FINGERPRINT
`manifest.json` stores a SHA-256 over every article id, its title and its body.
Change one word of one article and the fingerprint changes, so "is this index
stale?" becomes a string comparison instead of a guess. The manifest also
pins the embedding model and dimension — an index built with a 384-dimension
model is meaningless to a 768-dimension one, and recording it means the
mismatch is caught rather than silently producing garbage rankings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone

from app.config import Settings, get_settings
from app.database import db
from app.rag.chunking import chunk_articles
from app.rag.embeddings import EmbeddingUnavailable, get_embedding_provider
from app.rag.hybrid import CHUNKS_FILE, reset_retriever_cache
from app.rag.models import Chunk
from app.rag.vector_store import create_index

logger = logging.getLogger("resolveai.rag.ingest")

MANIFEST_FILE = "manifest.json"


def corpus_fingerprint(articles: list[dict]) -> str:
    """
    A SHA-256 over the knowledge base's content, used to detect staleness.

    Articles are hashed in id order, so the fingerprint depends on the content
    and not on the order rows happened to come back from SQLite.
    """
    digest = hashlib.sha256()
    for article in sorted(articles, key=lambda a: a["id"]):
        digest.update(article["id"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update((article.get("title") or "").encode("utf-8"))
        digest.update(b"\x00")
        digest.update((article.get("body") or "").encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def write_chunk_store(chunks: list[Chunk], settings: Settings) -> None:
    """Persist the chunk store — the text the index rows correspond to."""
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    (settings.index_dir / CHUNKS_FILE).write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8")


def read_manifest(settings: Settings) -> dict | None:
    """Return the existing manifest, or None if no index has been built."""
    path = settings.index_dir / MANIFEST_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def ingest(settings: Settings | None = None, dry_run: bool = False) -> dict:
    """
    Rebuild the retrieval index from `kb_articles`. Returns a summary dict.

    Idempotent: running it twice on an unchanged knowledge base produces
    byte-identical chunks and the same fingerprint, because chunking is
    deterministic and articles are read in id order.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    articles = db.all_kb_articles()
    if not articles:
        raise RuntimeError(
            "no rows in kb_articles — seed the database first: python -m app.database.seed")

    chunks = chunk_articles(articles, settings.chunk_size_words,
                            settings.chunk_overlap_words)
    fingerprint = corpus_fingerprint(articles)

    summary = {
        "articles": len(articles),
        "chunks": len(chunks),
        "avg_chunk_words": round(
            sum(len(c.text.split()) for c in chunks) / len(chunks), 1) if chunks else 0.0,
        "chunk_size_words": settings.chunk_size_words,
        "chunk_overlap_words": settings.chunk_overlap_words,
        "corpus_fingerprint": fingerprint,
        "dry_run": dry_run,
    }

    if dry_run:
        summary["seconds"] = round(time.perf_counter() - started, 2)
        return summary

    embedder = get_embedding_provider(settings)   # raises if unusable — build must not guess
    vectors = embedder.embed_documents([c.text for c in chunks])

    index = create_index(settings.vector_backend, embedder.dimension)
    index.build(vectors, [c.chunk_id for c in chunks])

    write_chunk_store(chunks, settings)
    index.save(settings.index_dir)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "embedding_provider": embedder.name,
        "embedding_model": embedder.model_id,
        "dimension": embedder.dimension,
        "vector_backend": index.backend,
        "articles": len(articles),
        "chunks": len(chunks),
        "chunk_size_words": settings.chunk_size_words,
        "chunk_overlap_words": settings.chunk_overlap_words,
        "corpus_fingerprint": fingerprint,
    }
    (settings.index_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    # A cached retriever now holds the OLD index — drop it.
    reset_retriever_cache()

    summary.update({
        "embedding_provider": embedder.name,
        "embedding_model": embedder.model_id,
        "dimension": embedder.dimension,
        "vector_backend": index.backend,
        "index_dir": str(settings.index_dir),
        "seconds": round(time.perf_counter() - started, 2),
    })
    return summary


def index_stats(settings: Settings | None = None) -> dict:
    """Report on the index currently on disk, including whether it is stale."""
    settings = settings or get_settings()
    manifest = read_manifest(settings)
    if manifest is None:
        return {"exists": False, "index_dir": str(settings.index_dir)}

    current = corpus_fingerprint(db.all_kb_articles())
    return {
        "exists": True,
        "index_dir": str(settings.index_dir),
        "stale": manifest.get("corpus_fingerprint") != current,
        **manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the ResolveAI-RAG retrieval index from kb_articles.")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild even when the fingerprint is unchanged")
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk and report without embedding or writing")
    parser.add_argument("--stats", action="store_true",
                        help="show the current index manifest and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    settings = get_settings()

    if args.stats:
        print(json.dumps(index_stats(settings), indent=2))
        return 0

    if not args.rebuild and not args.dry_run:
        manifest = read_manifest(settings)
        if manifest and manifest.get("corpus_fingerprint") == corpus_fingerprint(
                db.all_kb_articles()):
            print("index is already up to date (corpus fingerprint unchanged); "
                  "use --rebuild to force")
            print(json.dumps(index_stats(settings), indent=2))
            return 0

    try:
        summary = ingest(settings, dry_run=args.dry_run)
    except EmbeddingUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
