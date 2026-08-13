"""
conftest.py — shared pytest fixtures.

TWO RULES THIS FILE ENFORCES FOR THE WHOLE SUITE

1. NO CLOUD APIS. `MODEL_PROVIDER=mock` means no LLM key is ever needed.

2. NO HEAVY MODELS BY DEFAULT. `EMBEDDING_PROVIDER=hashing` means the suite
   needs neither PyTorch nor a 90 MB download, so it runs in seconds on a cold
   CI machine. The handful of tests that genuinely need real semantics are
   marked `@pytest.mark.slow` and opt in explicitly.

Every test gets its OWN temporary database and index directory, so tests
cannot see each other's state and can run in any order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `app` importable when pytest is invoked from the repository root.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings                       # noqa: E402
from app.database import db, seed as seed_module          # noqa: E402
from app.rag.hybrid import HybridRetriever, reset_retriever_cache  # noqa: E402
from app.rag.ingest import ingest                         # noqa: E402


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """
    Point every test at a throwaway database and index directory.

    `autouse=True` means no test can forget to do this and accidentally write
    to the developer's real `resolveai.db`.
    """
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("VECTOR_INDEX_PATH", str(tmp_path / "index"))
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    monkeypatch.setenv("MOCK_FAILURE_MODE", "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("HASHING_EMBEDDING_DIM", "256")
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_MODE", "hybrid")
    # Matches the shipped default, so the suite exercises the real code path.
    # Reranking is covered directly in test_reranker.py and opted into where a
    # test needs it.
    monkeypatch.setenv("RERANKER", "none")
    reset_retriever_cache()
    yield
    reset_retriever_cache()


@pytest.fixture
def seeded_db():
    """A fresh database with the full demo dataset."""
    return seed_module.seed()


@pytest.fixture
def settings(seeded_db):
    return get_settings()


@pytest.fixture
def built_index(settings):
    """A seeded database WITH the retrieval index built (hashing embeddings)."""
    ingest(settings)
    reset_retriever_cache()
    return settings


@pytest.fixture
def retriever(built_index):
    """A HybridRetriever with both arms available."""
    return HybridRetriever(built_index)


@pytest.fixture
def sample_articles():
    """
    A tiny hand-written corpus for unit tests.

    Small and fully known, so an assertion like "the ERR-9001 query must return
    article a2" is a statement about the retriever, not a lucky coincidence in
    a large corpus.
    """
    return [
        {
            "id": "a1",
            "title": "Refund policy",
            "body": ("Refunds are available within thirty days.\n\n"
                     "## Eligibility\n"
                     "An account must be in good standing to receive a refund. "
                     "Past due accounts are not eligible.\n\n"
                     "## How to request\n"
                     "Contact support with the invoice number and a reason."),
            "tags": "refund billing policy",
            "url": "https://docs.example.com/refunds",
            "product_area": "billing",
        },
        {
            "id": "a2",
            "title": "API error codes",
            "body": ("The API returns structured error codes.\n\n"
                     "## Rate limiting\n"
                     "Error ERR-9001 means you exceeded the request quota. "
                     "Back off and retry after the window resets.\n\n"
                     "## Authentication\n"
                     "Error ERR-9002 means the API key is invalid or deleted."),
            "tags": "api errors ERR-9001 ERR-9002 ratelimit",
            "url": "https://docs.example.com/errors",
            "product_area": "api",
        },
        {
            "id": "a3",
            "title": "Managing team members",
            "body": ("## Removing someone\n"
                     "When an employee departs, deactivate their seat from the "
                     "Members page. Billing adjusts on the next invoice."),
            "tags": "seats members team",
            "url": "https://docs.example.com/seats",
            "product_area": "account",
        },
    ]


@pytest.fixture
def sample_chunks(sample_articles):
    from app.rag.chunking import chunk_articles
    return chunk_articles(sample_articles, chunk_size_words=40, chunk_overlap_words=8)


@pytest.fixture
def ticket_factory(seeded_db):
    """Insert a ticket and return it as a dict."""
    def make(subject: str, body: str, sender: str = "priya@northwind.example",
             ticket_id: str | None = None) -> dict:
        ticket_id = ticket_id or db.new_id("tkt")
        account = db.query_one("SELECT id FROM accounts WHERE contact_email=?", (sender,))
        db.execute(
            """INSERT INTO tickets
               (id, account_id, sender_email, subject, body, channel, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'email', 'new', ?)""",
            (ticket_id, account["id"] if account else None, sender, subject, body,
             db.now_iso()))
        return dict(db.query_one("SELECT * FROM tickets WHERE id=?", (ticket_id,)))
    return make
