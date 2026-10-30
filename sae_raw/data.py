"""Data preparation: GPT-2 embedding table, Wikipedia token counts, diagnostics.

Deliberately avoids torch and transformers so that data preparation runs in a
light environment: the embedding table is read straight from the safetensors
file, and the GPT-2 tokenizer comes from the `tokenizers` library.
"""

import itertools
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

GPT2_REPO = "openai-community/gpt2"
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.en"

EMBEDDINGS_FILE = "embeddings.npy"
COUNTS_FILE = "token_counts.npy"
META_FILE = "meta.json"


# ----------------------------------------------------------------------------
# Embedding table
# ----------------------------------------------------------------------------

def extract_gpt2_embeddings(repo=GPT2_REPO):
    """Return (E, revision). E is the float32 input embedding table, shape [V, d].

    This is the token embedding matrix only: no positional embeddings and no
    transformer computation.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(repo, "model.safetensors")
    with safe_open(path, framework="np") as f:
        keys = [k for k in f.keys() if k.endswith("wte.weight")]
        if len(keys) != 1:
            raise RuntimeError(f"Expected one wte.weight tensor, found {keys}")
        E = f.get_tensor(keys[0]).astype(np.float32)
    revision = Path(path).parent.name  # snapshot directory is the commit hash
    return E, revision


# ----------------------------------------------------------------------------
# Token counts
# ----------------------------------------------------------------------------

def _count_batch(batch, encode_batch, counts):
    ids = encode_batch(batch)
    flat = np.fromiter(itertools.chain.from_iterable(ids), dtype=np.int64)
    counts += np.bincount(flat, minlength=len(counts))  # fails loudly if an id >= V
    return flat.size


def count_tokens(texts, encode_batch, vocab_size, max_tokens,
                 batch_size=1000, progress=None):
    """Count token occurrences over a stream of texts.

    encode_batch maps a list of strings to a list of token-id lists. Counting
    stops after the first batch that brings the total to max_tokens or more.
    Returns (counts, n_tokens, n_docs).
    """
    counts = np.zeros(vocab_size, dtype=np.int64)
    n_tokens = n_docs = 0
    batch = []
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            n_tokens += _count_batch(batch, encode_batch, counts)
            n_docs += len(batch)
            batch = []
            if progress is not None:
                progress(n_docs, n_tokens)
            if n_tokens >= max_tokens:
                break
    else:
        if batch:
            n_tokens += _count_batch(batch, encode_batch, counts)
            n_docs += len(batch)
    return counts, n_tokens, n_docs


def _wikipedia_row_group_index():
    """List (shard path, row group number) for every row group of the snapshot."""
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    shards = sorted(f for f in HfApi().list_repo_files(WIKI_DATASET, repo_type="dataset")
                    if f.startswith(WIKI_CONFIG + "/") and f.endswith(".parquet"))

    def n_row_groups(shard):
        with HfFileSystem().open(f"datasets/{WIKI_DATASET}/{shard}", "rb") as f:
            return pq.ParquetFile(f).metadata.num_row_groups

    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(n_row_groups, shards))
    return [(s, g) for s, n in zip(shards, sizes) for g in range(n)]


def wikipedia_texts(seed=0):
    """Yield English Wikipedia articles from randomly ordered parquet row groups.

    A row group holds about 1,000 articles. Row groups from all shards are
    visited in a seeded random order, so a partial pass is a sample of the
    whole snapshot rather than its first shard. Articles within a row group
    keep their stored order. Only the row groups actually used are downloaded.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    index = _wikipedia_row_group_index()
    fs = HfFileSystem()
    for k in np.random.default_rng(seed).permutation(len(index)):
        shard, group = index[k]
        with fs.open(f"datasets/{WIKI_DATASET}/{shard}", "rb") as f:
            table = pq.ParquetFile(f).read_row_group(group, columns=["text"])
        yield from table.column("text").to_pylist()


def gpt2_encode_batch(repo=GPT2_REPO):
    """Return an encode_batch function using the GPT-2 tokenizer."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(repo)
    return lambda texts: [e.ids for e in tok.encode_batch(texts)]


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

def embedding_diagnostics(E, block=1024):
    """Norm summary and nearest-neighbour cosine similarity for each token.

    Nearest-neighbour cosines are computed blockwise, so the full V x V
    similarity matrix is never held in memory. Two tokens on the same positive
    ray have cosine 1, so a cosine close to 1 flags a near-duplicate direction.
    """
    V = E.shape[0]
    norms = np.linalg.norm(E, axis=1)
    U = E / np.maximum(norms, 1e-12)[:, None]
    nn_cos = np.empty(V, dtype=np.float32)
    for start in range(0, V, block):
        S = U[start:start + block] @ U.T
        rows = np.arange(S.shape[0])
        S[rows, start + rows] = -np.inf  # exclude self-similarity
        nn_cos[start:start + block] = S.max(axis=1)
    return {
        "vocab_size": int(V),
        "dim": int(E.shape[1]),
        "n_zero_norm": int((norms == 0).sum()),
        "norm_min": float(norms.min()),
        "norm_median": float(np.median(norms)),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
        "nn_cos_median": float(np.median(nn_cos)),
        "nn_cos_p99": float(np.percentile(nn_cos, 99)),
        "nn_cos_max": float(nn_cos.max()),
        "n_nn_cos_above_0.99": int((nn_cos > 0.99).sum()),
        "n_nn_cos_above_0.999": int((nn_cos > 0.999).sum()),
    }


def count_diagnostics(counts):
    """Summary of how concentrated the token-frequency distribution is."""
    total = counts.sum()
    sorted_counts = np.sort(counts)[::-1]
    cum = np.cumsum(sorted_counts) / total
    return {
        "n_tokens_counted": int(total),
        "n_zero_count": int((counts == 0).sum()),
        "n_count_below_10": int((counts < 10).sum()),
        "mass_top_1000": float(cum[999]),
        "mass_top_10000": float(cum[9999]),
        "n_tokens_for_90pct_mass": int(np.searchsorted(cum, 0.9) + 1),
        "n_tokens_for_99pct_mass": int(np.searchsorted(cum, 0.99) + 1),
    }


def select_subset(E, counts, n):
    """Rows of the n most frequent tokens: the subvocabularies of the experiments.

    Returns (E_sub, counts_sub, token_ids), ordered from most to least frequent.
    A subset run uses dictionary size m = n. Ties keep the vocabulary order.
    """
    ids = np.argsort(-counts, kind="stable")[:n]
    return E[ids], counts[ids], ids


# ----------------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------------

def is_prepared(data_dir="data"):
    data_dir = Path(data_dir)
    return all((data_dir / f).exists() for f in (EMBEDDINGS_FILE, COUNTS_FILE, META_FILE))


def prepare(data_dir="data", max_tokens=200_000_000, batch_size=1000, seed=0, recount=False,
            log=print):
    """Extract the embedding table, count Wikipedia tokens and save both with diagnostics.

    Writes embeddings.npy, token_counts.npy and meta.json to data_dir. The
    repository ships token_counts.npy and meta.json, so by default only the
    embedding table is downloaded and the shipped counts are kept. With
    recount=True, or if the counts are missing, Wikipedia is counted again.
    """
    data_dir = Path(data_dir)
    log("Extracting GPT-2 embedding table ...")
    E, revision = extract_gpt2_embeddings()
    log(f"  shape {E.shape}, dtype {E.dtype}, revision {revision}")
    emb_diag = embedding_diagnostics(E)
    for k, v in emb_diag.items():
        log(f"  {k}: {v}")

    if not recount and (data_dir / COUNTS_FILE).exists() and (data_dir / META_FILE).exists():
        meta = json.loads((data_dir / META_FILE).read_text())
        if meta.get("gpt2_revision") != revision:
            log(f"  note: the token counts were made with GPT-2 revision {meta.get('gpt2_revision')}")
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / EMBEDDINGS_FILE, E)
        log(f"Saved the embedding table to {data_dir}/ and kept the existing token counts.")
        return

    log(f"Counting tokens in Wikipedia (up to {max_tokens:,}) ...")
    t0 = time.time()

    def progress(n_docs, n_tokens):
        if (n_docs // batch_size) % 20 == 0:
            log(f"  {n_docs:,} articles, {n_tokens:,} tokens "
                f"({n_tokens / (time.time() - t0):,.0f} tok/s)")

    counts, n_tokens, n_docs = count_tokens(wikipedia_texts(seed=seed), gpt2_encode_batch(),
                                            E.shape[0], max_tokens, batch_size, progress)
    log(f"  done: {n_docs:,} articles, {n_tokens:,} tokens in {time.time() - t0:.0f}s")
    cnt_diag = count_diagnostics(counts)
    for k, v in cnt_diag.items():
        log(f"  {k}: {v}")

    meta = {
        "gpt2_repo": GPT2_REPO,
        "gpt2_revision": revision,
        "wikipedia_dataset": WIKI_DATASET,
        "wikipedia_config": WIKI_CONFIG,
        "wikipedia_sample_seed": seed,
        "wikipedia_sampling": "random row groups (about 1000 articles each) across all shards",
        "n_articles_counted": n_docs,
        "embedding_diagnostics": emb_diag,
        "count_diagnostics": cnt_diag,
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in
                      ["numpy", "safetensors", "tokenizers", "pyarrow", "huggingface_hub"]},
    }
    save_prepared(data_dir, E, counts, meta)
    log(f"Saved to {data_dir}/")


def save_prepared(data_dir, E, counts, meta):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / EMBEDDINGS_FILE, E)
    np.save(data_dir / COUNTS_FILE, counts)
    (data_dir / META_FILE).write_text(json.dumps(meta, indent=2))


def load_prepared(data_dir="data", subset=0):
    """Return (E, counts) as saved by scripts/prepare_data.py.

    If subset > 0, return only the rows of the subset most frequent tokens
    (see select_subset), as used by --subset runs.
    """
    data_dir = Path(data_dir)
    E, counts = np.load(data_dir / EMBEDDINGS_FILE), np.load(data_dir / COUNTS_FILE)
    if subset > 0:
        E, counts, _ = select_subset(E, counts, subset)
    return E, counts
