"""Experiment 5: share of TopK features alive at initialisation, by vocabulary size and k.

One command runs everything:

    python experiments/exp5_alive_at_init/run.py

For each vocabulary size V (the V most frequent tokens, as for --subset; the
full table for V = 50,257), it builds the randomly initialised TopK SAE with
m = V for each seed, and counts the features alive at initialisation for
k = 1, 2, 4, ..., 256. A feature is alive if, for at least one token, it is
among the k largest pre-activations with a positive pre-activation. No
training is involved. The top-256 pre-activations of each token give the
top-k sets for every smaller k, so each (V, seed) needs one pass.

Results go to runs/exp5/alive_at_init.csv (one row per V, seed and k) and are
printed as a table of the mean share over the seeds.
"""

import argparse
import csv
from pathlib import Path

import torch

from sae_raw import data
from sae_raw.model import build_model

SIZES = [2000, 5000, 10000, 20000, 30000, 40000, 50257]
KS = [2 ** i for i in range(9)]  # 1 to 256
SEEDS = [0, 1]


@torch.no_grad()
def alive_counts(E, seed, ks, chunk=1024):
    """Number of features alive at initialisation for each k in ks (m = V)."""
    V, d = E.shape
    model = build_model("topk", d, V, seed=seed, k=max(ks))
    kmax = max(ks)
    alive = {k: torch.zeros(V, dtype=torch.bool) for k in ks}
    for start in range(0, V, chunk):
        a = E[start:start + chunk] @ model.W_enc.T + model.b_enc
        vals, idx = a.topk(kmax, dim=1)  # sorted, largest first
        for k in ks:
            on = vals[:, :k] > 0
            alive[k][idx[:, :k][on]] = True
    return {k: int(alive[k].sum()) for k in ks}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/exp5")
    p.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args(argv)
    if not data.is_prepared(args.data_dir):
        data.prepare(args.data_dir)
    rows = []
    for V in args.sizes:
        E_np, _ = data.load_prepared(args.data_dir, subset=V)
        E = torch.as_tensor(E_np, dtype=torch.float32)
        for seed in args.seeds:
            counts = alive_counts(E, seed, KS)
            for k in KS:
                rows.append({"V": len(E), "seed": seed, "k": k, "n_alive": counts[k],
                             "share": counts[k] / len(E)})
            print(f"V = {len(E):,}, seed {seed}: " +
                  ", ".join(f"k={k} {100 * counts[k] / len(E):.1f}%" for k in KS), flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "alive_at_init.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nMean share alive at initialisation over the seeds (%):")
    print("V \\ k  " + "".join(f"{k:>7}" for k in KS))
    for V in sorted({r["V"] for r in rows}):
        means = [100 * sum(r["share"] for r in rows if r["V"] == V and r["k"] == k) /
                 len(args.seeds) for k in KS]
        print(f"{V:>6} " + "".join(f"{m:7.1f}" for m in means))
    return rows


if __name__ == "__main__":
    main()
