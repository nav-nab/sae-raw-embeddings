import numpy as np

from sae_raw import data


def fake_encode_batch(texts):
    """Each text is a space-separated list of integer token ids."""
    return [[int(t) for t in text.split()] for text in texts]


def test_count_tokens_counts_everything_and_flushes_remainder():
    texts = ["0 1 1", "2 2 2", "3", "1 0"]
    counts, n_tokens, n_docs = data.count_tokens(
        texts, fake_encode_batch, vocab_size=5, max_tokens=10**9, batch_size=3)
    assert counts.tolist() == [2, 3, 3, 1, 0]
    assert (n_tokens, n_docs) == (9, 4)


def test_count_tokens_stops_after_max_tokens():
    texts = ["0 1"] * 100
    counts, n_tokens, n_docs = data.count_tokens(
        texts, fake_encode_batch, vocab_size=2, max_tokens=10, batch_size=2)
    # Each batch holds 4 tokens, so counting stops after the third batch (12 >= 10).
    assert n_tokens == 12 and n_docs == 6
    assert counts.tolist() == [6, 6]


def test_embedding_diagnostics_flags_duplicate_direction():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(50, 16)).astype(np.float32)
    E[5] = 2.0 * E[3]  # same positive ray, different norm
    diag = data.embedding_diagnostics(E, block=7)  # block does not divide V
    assert diag["n_nn_cos_above_0.999"] == 2
    assert diag["nn_cos_max"] > 0.999
    assert diag["n_zero_norm"] == 0


def test_embedding_diagnostics_matches_dense_computation():
    rng = np.random.default_rng(1)
    E = rng.normal(size=(40, 8)).astype(np.float32)
    U = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    diag = data.embedding_diagnostics(E, block=9)
    assert np.isclose(diag["nn_cos_max"], S.max(axis=1).max(), atol=1e-5)
    assert np.isclose(diag["nn_cos_median"], np.median(S.max(axis=1)), atol=1e-5)


def test_count_diagnostics():
    counts = np.array([0] * 3 + [5] * 2 + [1000] * 1 + [1] * 14, dtype=np.int64)
    counts = np.concatenate([counts, np.zeros(10000, dtype=np.int64)])
    diag = data.count_diagnostics(counts)
    assert diag["n_zero_count"] == 3 + 10000
    assert diag["n_count_below_10"] == 3 + 14 + 2 + 10000
    assert diag["n_tokens_counted"] == 1024
    assert diag["n_tokens_for_90pct_mass"] == 1


def test_save_and_load_roundtrip(tmp_path):
    E = np.arange(12, dtype=np.float32).reshape(4, 3)
    counts = np.array([1, 2, 3, 4], dtype=np.int64)
    data.save_prepared(tmp_path, E, counts, {"note": "x"})
    E2, counts2 = data.load_prepared(tmp_path)
    assert np.array_equal(E, E2) and np.array_equal(counts, counts2)


def test_prepare_keeps_shipped_counts(tmp_path, monkeypatch):
    E = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    counts = np.array([5, 0, 7, 1], dtype=np.int64)
    data.save_prepared(tmp_path, np.zeros((1, 1), dtype=np.float32), counts, {"gpt2_revision": "r"})
    (tmp_path / data.EMBEDDINGS_FILE).unlink()
    monkeypatch.setattr(data, "extract_gpt2_embeddings", lambda: (E, "r"))
    monkeypatch.setattr(data, "wikipedia_texts",
                        lambda seed=0: (_ for _ in ()).throw(AssertionError("should not count")))
    data.prepare(tmp_path, log=lambda msg: None)
    E2, counts2 = data.load_prepared(tmp_path)
    assert np.array_equal(E, E2) and np.array_equal(counts, counts2)
