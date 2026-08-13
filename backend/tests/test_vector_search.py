"""Tests for the vector index: both backends, persistence, and metadata mapping."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.rag.embeddings import HashingEmbeddings, l2_normalize
from app.rag.vector_store import (FaissVectorIndex, NumpyVectorIndex,
                                  VectorIndexUnavailable, create_index,
                                  load_index)

BACKENDS = ["numpy", "faiss"]


def _vectors(texts: list[str], dimension: int = 64) -> np.ndarray:
    return HashingEmbeddings(dimension).embed_documents(texts)


def _build(backend: str, texts: list[str], ids: list[str], dimension: int = 64):
    index = create_index(backend, dimension)
    index.build(_vectors(texts, dimension), ids)
    return index


def test_search_returns_chunk_ids_not_row_numbers():
    """The interface hides index internals — callers never see a row position."""
    index = _build("numpy", ["refund policy", "api error codes"], ["c1", "c2"])
    results = index.search(HashingEmbeddings(64).embed_text("refund"), k=2)
    assert all(isinstance(chunk_id, str) for chunk_id, _ in results)
    assert {chunk_id for chunk_id, _ in results} == {"c1", "c2"}


def test_nearest_vector_ranks_first():
    texts = ["refund policy and eligibility", "api rate limit error codes",
             "saml single sign-on setup"]
    index = _build("numpy", texts, ["c1", "c2", "c3"])
    top = index.search(HashingEmbeddings(64).embed_text("refund eligibility"), k=1)
    assert top[0][0] == "c1"


def test_results_are_ordered_by_descending_similarity():
    index = _build("numpy", ["alpha beta", "beta gamma", "delta"], ["a", "b", "c"])
    scores = [s for _, s in index.search(HashingEmbeddings(64).embed_text("alpha"), k=3)]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_the_corpus_is_safe():
    index = _build("numpy", ["one", "two"], ["a", "b"])
    assert len(index.search(HashingEmbeddings(64).embed_text("one"), k=50)) == 2


def test_searching_an_empty_index_returns_nothing():
    assert NumpyVectorIndex(8).search(np.zeros(8, dtype=np.float32), k=3) == []


def test_dimension_mismatch_is_rejected():
    """A 384-dim index fed 64-dim vectors must fail loudly, not rank garbage."""
    with pytest.raises(ValueError, match="dimension"):
        NumpyVectorIndex(384).build(_vectors(["a"], 64), ["a"])


def test_vector_and_id_count_must_agree():
    with pytest.raises(ValueError, match="chunk ids"):
        NumpyVectorIndex(64).build(_vectors(["a", "b"], 64), ["only-one"])


def test_inner_product_of_normalised_vectors_is_cosine():
    """The property the whole design rests on: normalise once, IP == cosine."""
    a = l2_normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = l2_normalize(np.array([2.0, 4.0, 6.0], dtype=np.float32))
    assert np.isclose(float(a @ b), 1.0, atol=1e-6)   # same direction


@pytest.mark.parametrize("backend", BACKENDS)
def test_round_trip_through_disk_preserves_results(backend, tmp_path):
    texts = ["refund policy", "api error codes", "seat management"]
    ids = ["c1", "c2", "c3"]
    original = _build(backend, texts, ids)
    query = HashingEmbeddings(64).embed_text("api errors")
    before = original.search(query, k=3)

    original.save(tmp_path)
    reloaded = load_index(tmp_path, backend)

    assert reloaded.chunk_ids == ids            # the mapping survived
    after = reloaded.search(query, k=3)
    assert [c for c, _ in before] == [c for c, _ in after]


@pytest.mark.parametrize("backend", BACKENDS)
def test_saving_writes_the_expected_artifacts(backend, tmp_path):
    _build(backend, ["one", "two"], ["a", "b"]).save(tmp_path)
    assert (tmp_path / "vectors.npy").exists()
    assert (tmp_path / "ids.json").exists()
    if backend == "faiss":
        assert (tmp_path / "index.faiss").exists()


def test_loading_a_missing_index_raises_the_typed_error(tmp_path):
    """The retriever catches exactly this to trigger the lexical fallback."""
    with pytest.raises(VectorIndexUnavailable):
        load_index(tmp_path / "does-not-exist", "numpy")


def test_a_mismatch_between_vectors_and_ids_is_detected(tmp_path):
    """A corrupt index must be rejected, not silently return the wrong chunk."""
    _build("numpy", ["one", "two"], ["a", "b"]).save(tmp_path)
    (tmp_path / "ids.json").write_text(json.dumps(["a"]), encoding="utf-8")
    with pytest.raises(VectorIndexUnavailable, match="mismatch"):
        load_index(tmp_path, "numpy")


def test_both_backends_produce_identical_rankings():
    """
    FAISS is an optimisation, not a different algorithm. If these ever diverge,
    the numpy fallback would silently change retrieval quality.
    """
    pytest.importorskip("faiss")
    texts = ["refund policy eligibility", "api rate limit ERR-9001",
             "saml sso setup", "csv export encoding"]
    ids = ["c1", "c2", "c3", "c4"]
    query = HashingEmbeddings(64).embed_text("rate limit error")

    numpy_result = _build("numpy", texts, ids).search(query, k=4)
    faiss_result = _build("faiss", texts, ids).search(query, k=4)

    assert [c for c, _ in numpy_result] == [c for c, _ in faiss_result]
    for (_, a), (_, b) in zip(numpy_result, faiss_result):
        assert np.isclose(a, b, atol=1e-5)


def test_create_index_falls_back_to_numpy_when_faiss_is_missing(monkeypatch):
    """
    This fallback is safe (identical results) — unlike the embedding fallback,
    which changes what the system can actually do and therefore raises instead.
    """
    monkeypatch.setattr(FaissVectorIndex, "_import_faiss",
                        staticmethod(lambda: (_ for _ in ()).throw(
                            VectorIndexUnavailable("faiss missing"))))
    assert isinstance(create_index("faiss", 16), NumpyVectorIndex)
