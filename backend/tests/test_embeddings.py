"""Tests for the embedding provider interface and both implementations."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import get_settings
from app.rag.embeddings import (EmbeddingUnavailable, HashingEmbeddings,
                                SentenceTransformerEmbeddings,
                                get_embedding_provider, l2_normalize)


def test_l2_normalize_handles_a_single_vector():
    """embed_text returns 1-D; embed_documents returns 2-D. Both must work."""
    result = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert result.shape == (2,)
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_l2_normalize_handles_a_batch():
    result = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    assert result.shape == (2, 2)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_l2_normalize_leaves_a_zero_vector_alone():
    """A zero vector must not become NaN and poison every later dot product."""
    result = l2_normalize(np.zeros(4, dtype=np.float32))
    assert not np.isnan(result).any()


def test_hashing_provider_reports_its_dimension():
    provider = HashingEmbeddings(128)
    assert provider.dimension == 128
    assert provider.name == "hashing"
    assert provider.embed_text("hello world").shape == (128,)


def test_embed_documents_returns_one_row_per_text():
    provider = HashingEmbeddings(64)
    vectors = provider.embed_documents(["alpha", "beta", "gamma"])
    assert vectors.shape == (3, 64)


def test_embeddings_are_unit_length():
    """Unit length is what makes an inner-product index compute cosine."""
    vectors = HashingEmbeddings(64).embed_documents(["alpha beta", "gamma delta"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_hashing_is_deterministic_across_instances():
    """
    SHA-256, not Python's hash(): built-in string hashing is randomised per
    process, which would make an index unreproducible between runs.
    """
    first = HashingEmbeddings(64).embed_text("scheduled report delivery")
    second = HashingEmbeddings(64).embed_text("scheduled report delivery")
    assert np.array_equal(first, second)


def test_hashing_scores_shared_vocabulary_higher():
    provider = HashingEmbeddings(512)
    query = provider.embed_text("export data to csv")
    related = provider.embed_text("exporting your data as a csv file")
    unrelated = provider.embed_text("single sign-on with SAML identity providers")
    assert float(query @ related) > float(query @ unrelated)


def test_empty_document_list_returns_an_empty_matrix():
    vectors = HashingEmbeddings(32).embed_documents([])
    assert vectors.shape == (0, 32)


def test_factory_returns_the_configured_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("HASHING_EMBEDDING_DIM", "77")
    provider = get_embedding_provider(get_settings())
    assert isinstance(provider, HashingEmbeddings)
    assert provider.dimension == 77


def test_factory_raises_on_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "not-a-real-provider")
    with pytest.raises(EmbeddingUnavailable):
        get_embedding_provider(get_settings())


def test_factory_raises_when_sentence_transformers_is_missing(monkeypatch):
    """
    The factory must RAISE rather than silently substitute a weaker model.

    Quietly swapping in hashing embeddings would let the system keep reporting
    "hybrid" while doing no semantic retrieval at all.
    """
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(EmbeddingUnavailable, match="not installed"):
        get_embedding_provider(get_settings())


def test_sentence_transformer_model_is_not_loaded_at_construction():
    """Lazy loading: constructing must not pull in PyTorch or touch the disk."""
    provider = SentenceTransformerEmbeddings("some/model-id")
    assert provider._model is None
    assert provider.model_id == "some/model-id"


@pytest.mark.slow
def test_real_model_captures_meaning_beyond_shared_words(monkeypatch):
    """
    The claim that justifies semantic retrieval, tested against the real model.

    Query and related sentence share almost no content words, yet must land
    closer together than an unrelated sentence does. The hashing provider
    CANNOT pass this — which is exactly why the docs call it non-semantic.
    """
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
    provider = get_embedding_provider(get_settings())

    query = provider.embed_text("someone left the company and we still pay for them")
    related = provider.embed_text("remove a seat from your team when a user departs")
    unrelated = provider.embed_text("reports can be delivered as PDF or CSV")

    assert float(query @ related) > float(query @ unrelated)
    assert provider.dimension == 384


@pytest.mark.slow
def test_real_model_has_documented_blind_spots(monkeypatch):
    """
    An HONEST test of a measured limitation, not a hoped-for result.

    all-MiniLM-L6-v2 does NOT reliably relate the bare noun "licences" to
    "seat". Measured on this machine, "we are paying for licences nobody uses"
    scores 0.014 against "deactivate a seat when an employee leaves" but 0.038
    against an unrelated sentence about CSV exports — the wrong way round.

    Add surrounding context ("someone left the company and we still pay for
    them") and the same model gets it right, which is the real lesson: a small
    embedding model needs context, and short keyword-ish queries are precisely
    where the LEXICAL arm has to carry the result. This test is why the system
    is hybrid rather than semantic-only.
    """
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
    provider = get_embedding_provider(get_settings())

    bare = provider.embed_text("we are paying for licences nobody uses")
    seat = provider.embed_text("deactivate a seat when an employee leaves")
    unrelated = provider.embed_text("export your table data to a CSV file")

    # Documenting the failure. If a future model fixes this, the test fails and
    # the docstring above must be updated — which is the point.
    assert float(bare @ seat) < float(bare @ unrelated)
