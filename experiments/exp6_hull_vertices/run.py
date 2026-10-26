"""Experiment 6: can a ReLU encoder output one-hot codes? (plan: docs/experiment_6_hull_vertices.md)

One command runs everything:

    python experiments/exp6_hull_vertices/run.py

A ReLU feature can fire on token j alone exactly when some hyperplane has x_j
strictly on one side and every other embedding on or behind it, that is, when
x_j is a vertex of the convex hull of the table (Appendix B of
docs/write_up_plan.md). Two tests decide this for every token:

1. The own-direction test (seconds). With w = x_j / |x_j|, token j passes if
   |x_j| exceeds the largest projection of any other token on w. Its margin
   is that gap as a fraction of |x_j|.
2. A linear programme for the tokens that fail (about 20 minutes on the full
   table). It maximises w.x_j + b subject to w.x_i + b <= 0 for all i != j and
   -1 <= w <= 1. Token j is separable iff the optimum is positive. Constraints
   are added in rounds, starting with the tokens that project furthest along
   x_j's direction.

If --exp2-run points to a finished Experiment 2 run, recovery there is
tabulated against the own-direction margin.

Results go to <out-dir>/hull_<subset or full>.csv, one row per token. A
progress line with an ETA is printed every minute during the linear programmes.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog

from sae_raw import data
from sae_raw.evaluation import recovery

FIRST_CONSTRAINTS = 2000  # constraints in the first round of each linear programme
ADD_PER_ROUND = 2000      # most violated constraints added per later round
TOL = 1e-9
MARGIN_BINS = [-np.inf, 0, 0.1, 0.2, 0.3, 0.4, np.inf]


@torch.no_grad()
def own_direction_margin(E, chunk=2048):
    """For each token: (|x_j| - max_{i != j} x_i . x_j/|x_j|) / |x_j|."""
    E = torch.as_tensor(E, dtype=torch.float32)
    n = E.norm(dim=1)
    U = E / n[:, None]
    worst = torch.empty(len(E))
    for s in range(0, len(E), chunk):
        P = U[s:s + chunk] @ E.T
        r = torch.arange(P.shape[0])
        P[r, r + s] = -float("inf")
        worst[s:s + chunk] = P.max(dim=1).values
    return ((n - worst) / n).numpy()


def separate(E, j):
    """Linear programme for token j. Returns (optimum, margin, rounds).

    The optimum is max w.x_j + b with the other tokens on or behind the
    hyperplane and |w_i| <= 1; it is positive iff x_j is a hull vertex. The
    margin is the gap to the nearest other token along w/|w|, as a fraction
    of |x_j|: a lower bound on the best possible margin.
    """
    V, d = E.shape
    others = np.delete(np.arange(V), j)
    xj = E[j]
    order = np.argsort(-(E[others] @ (xj / np.linalg.norm(xj))))
    active = list(others[order[:FIRST_CONSTRAINTS]])
    cost = np.concatenate([-xj, [-1.0]])
    bounds = [(-1, 1)] * d + [(None, None)]
    rounds = 0
    while True:
        rounds += 1
        A = np.hstack([E[active], np.ones((len(active), 1))])
        r = linprog(cost, A_ub=A, b_ub=np.zeros(len(active)), bounds=bounds, method="highs")
        w, b = r.x[:d], r.x[d]
        scores = E[others] @ w + b
        viol = others[scores > TOL]
        if len(viol) == 0:
            break
        active += list(viol[np.argsort(-scores[scores > TOL])[:ADD_PER_ROUND]])
    u = w / np.linalg.norm(w)
    margin = (xj @ u - (E[others] @ u).max()) / np.linalg.norm(xj)
    return -r.fun, float(margin), rounds


def recovered_mask(run_dir, E):
    """Tokens recovered at cosine > 0.9 by the model saved in run_dir."""
    W = torch.load(Path(run_dir) / "model.pt", map_location="cpu")["W_dec"]
    tok_cos, tok_feat, feat_tok = recovery(W, torch.as_tensor(E, dtype=torch.float32))
    mutual = feat_tok[tok_feat] == torch.arange(len(E))
    return (mutual & (tok_cos > 0.9)).numpy()


def margin_table(margin, rec):
    rows = []
    for lo, hi in zip(MARGIN_BINS[:-1], MARGIN_BINS[1:]):
        s = (margin > lo) & (margin <= hi)
        rows.append((lo, hi, int(s.sum()), float(rec[s].mean()) if s.any() else float("nan")))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/exp6")
    p.add_argument("--subset", type=int, default=0, help="most frequent tokens only (0 = full table)")
    p.add_argument("--exp2-run", default="runs/exp2",
                   help="Experiment 2 run directory, for the recovery table (skipped if absent)")
    args = p.parse_args(argv)
    if not data.is_prepared(args.data_dir):
        data.prepare(args.data_dir)
    E, counts = data.load_prepared(args.data_dir, subset=args.subset)
    E = E.astype(np.float64)
    V = len(E)

    margin = own_direction_margin(E)
    fails = np.where(margin <= 0)[0]
    print(f"Own-direction test: {V - len(fails):,} of {V:,} tokens pass, {len(fails):,} fail.", flush=True)

    lp_opt = np.full(V, np.nan)
    lp_margin = np.full(V, np.nan)
    t0 = last = time.time()
    for c, j in enumerate(fails):
        lp_opt[j], lp_margin[j], _ = separate(E, j)
        if time.time() - last > 60:
            last = time.time()
            eta = (last - t0) / (c + 1) * (len(fails) - c - 1)
            print(f"  linear programmes: {c + 1} of {len(fails)} done, ETA {eta / 60:.0f} min", flush=True)
    separable = (margin > 0) | (lp_opt > TOL)
    print(f"Linear programme: {int((lp_opt[fails] > TOL).sum())} of {len(fails)} failing tokens are separable.")
    print(f"Hull vertices: {int(separable.sum()):,} of {V:,}.")
    if len(fails):
        q = np.quantile(lp_margin[fails], [0, 0.5, 1])
        print(f"  margins from the linear programme: min {q[0]:.3f}, median {q[1]:.3f}, max {q[2]:.3f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"hull_{'subset' + str(args.subset) if args.subset else 'full'}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token_row", "count", "own_margin", "lp_optimum", "lp_margin", "separable"])
        for j in range(V):
            w.writerow([j, int(counts[j]), f"{margin[j]:.6f}", "" if np.isnan(lp_opt[j]) else f"{lp_opt[j]:.6g}",
                        "" if np.isnan(lp_margin[j]) else f"{lp_margin[j]:.6f}", int(separable[j])])
    print(f"Per-token results written to {path}.")

    if not args.subset and (Path(args.exp2_run) / "model.pt").exists():
        rec = recovered_mask(args.exp2_run, E)
        print(f"\nExperiment 2 recovery (cosine > 0.9) by own-direction margin, {int(rec.sum()):,} recovered:")
        for lo, hi, n, share in margin_table(margin, rec):
            print(f"  ({lo:>5}, {hi:>4}]: {n:>6,} tokens, {100 * share:5.1f}% recovered")
    return margin, lp_opt, lp_margin


if __name__ == "__main__":
    main()
