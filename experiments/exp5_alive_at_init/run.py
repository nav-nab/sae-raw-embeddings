"""Experiment 5: share of TopK features alive at initialisation, by vocabulary size and k.

One command runs everything:

    python experiments/exp5_alive_at_init/run.py                 # most frequent tokens
    python experiments/exp5_alive_at_init/run.py --order random  # random tokens (control)

For each vocabulary size V it takes V tokens of the embedding table and builds
the randomly initialised TopK SAE with m = V for each seed. It then counts the
features alive at initialisation for every k from 1 to 256. A feature is alive
if, for at least one token, it is among the k largest pre-activations with a
positive pre-activation. No training is involved.

--order freq (default) takes the V most frequent tokens, as for --subset, so
the vocabularies are nested. --order random takes the first V tokens of one
fixed random ordering of the vocabulary, so they are nested too. At V = 50,257
both are the whole table.

Each (V, seed) needs one pass: the rank at which each feature first appears,
with a positive pre-activation, in some token's top 256 gives its alive status
at every smaller k.

Results go to <out-dir>/alive_at_init_<order>.csv (one row per V, seed and k),
and a summary table of the mean over the seeds is printed.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from sae_raw import data
from sae_raw.model import build_model

SIZES = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 7500, 8500, 10000, 12500,
         15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000,
         42500, 45000, 50257]
K_MAX = 256
SEEDS = [0, 1]
ORDER_SEED = 1234  # seed of the random ordering used by --order random


@torch.no_grad()
def alive_counts(E, seed, ks, chunk=1024):
    """Number of features alive at initialisation for each k in ks (m = V)."""
    V, d = E.shape
    kmax = max(ks)
    model = build_model("topk", d, V, seed=seed, k=kmax)
    never = kmax + 1
    first = torch.full((V,), never, dtype=torch.long)  # best rank at which a feature is alive
    ranks = torch.arange(1, kmax + 1)
    for start in range(0, E.shape[0], chunk):
        a = E[start:start + chunk] @ model.W_enc.T + model.b_enc
        vals, idx = a.topk(kmax, dim=1)  # sorted, largest first
        r = torch.where(vals > 0, ranks.expand_as(idx), torch.full_like(idx, never))
        first.scatter_reduce_(0, idx.reshape(-1), r.reshape(-1), reduce="amin")
    alive_at = torch.bincount(first[first <= kmax], minlength=kmax + 1).cumsum(0)
    return {k: int(alive_at[k]) for k in ks}


def token_order(order, counts, data_dir):
    if order == "freq":
        return np.argsort(-counts, kind="stable")
    return np.random.default_rng(ORDER_SEED).permutation(len(counts))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/exp5")
    p.add_argument("--order", choices=["freq", "random"], default="freq")
    p.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args(argv)
    if not data.is_prepared(args.data_dir):
        data.prepare(args.data_dir)
    E_all, counts = data.load_prepared(args.data_dir)
    order = token_order(args.order, counts, args.data_dir)
    ks = list(range(1, K_MAX + 1))
    rows = []
    for V in args.sizes:
        E = torch.as_tensor(E_all[order[:V]], dtype=torch.float32)
        for seed in args.seeds:
            n = alive_counts(E, seed, ks)
            rows += [{"V": V, "order": args.order, "seed": seed, "k": k, "n_alive": n[k],
                      "share": n[k] / V} for k in ks]
        mean32 = np.mean([r["share"] for r in rows if r["V"] == V and r["k"] == 32])
        print(f"V = {V:,}: {100 * mean32:.1f}% alive at k = 32 (mean over seeds)", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"alive_at_init_{args.order}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean = {(V, k): np.mean([r["share"] for r in rows if r["V"] == V and r["k"] == k])
            for V in args.sizes for k in ks}
    print(f"\nOrder {args.order}. Smallest k with the mean share alive at or above each target:")
    print("     V   90%   93%   95%   97%")
    for V in args.sizes:
        thr = [next((k for k in ks if mean[V, k] >= t), None) for t in (0.90, 0.93, 0.95, 0.97)]
        print(f"{V:>6} " + "".join(f"{t:>6}" if t else "     -" for t in thr))
    return rows


if __name__ == "__main__":
    main()
