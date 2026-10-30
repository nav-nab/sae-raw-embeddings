"""How many features each token uses in the best ReLU SAEs, with and without a cutoff.

    python scripts/code_spread.py

For the best Experiment 4 run at each lambda (on the 2,000 most frequent
tokens) and the full-scale Experiment 2 run, it loads the final model.pt and
reports, over all tokens:

- the number of active features per token (positive coefficient): mean,
  median, maximum, and the share of tokens using 1, 2, 3 and 4 or more;
- the same after a cutoff that ignores features whose coefficient is below 10%
  of the token's largest coefficient;
- for tokens with two or more active features, the second-largest coefficient
  as a share of the largest (median, and share below 10%), and the median of
  all secondary coefficients as a share of the largest.

These numbers are the code-spread table, the cutoff analysis and the
second-coefficient figures of the paper (Section 4.3, Appendix E.2, Table 7).
They need the five final models, which are kept under runs/ but not tracked by
git; rerun the experiments to regenerate them.
"""

import argparse
from pathlib import Path

import torch

from sae_raw import data
from sae_raw.model import ReLUSAE

CUTOFF = 0.1
# (label, run directory, L1 coefficient, vocabulary size or 0 for the full table)
RUNS = [
    ("lambda = 0.03, 2k subset", "exp4/lam0.03_lr1_b256", 0.03, 2000),
    ("lambda = 0.1, 2k subset", "exp4/lam0.1_lr1_b1024", 0.1, 2000),
    ("lambda = 0.3, 2k subset", "exp4/lam0.3_lr1_b256", 0.3, 2000),
    ("lambda = 1, 2k subset", "exp4/lam1_lr0.25_b256", 1.0, 2000),
    ("lambda = 0.3, full table", "exp2", 0.3, 0),
]


def shares(l0):
    l0 = l0.float()
    return {"mean": float(l0.mean()), "median": float(l0.median()), "max": int(l0.max()),
            "1": float((l0 == 1).float().mean()), "2": float((l0 == 2).float().mean()),
            "3": float((l0 == 3).float().mean()), "4+": float((l0 >= 4).float().mean())}


@torch.no_grad()
def spread(model, E, chunk=256):
    l0, l0_cut, second, secondary = [], [], [], []
    for start in range(0, E.shape[0], chunk):
        f = model(E[start:start + chunk])[1]
        top2 = f.topk(2, dim=1).values
        f_max = top2[:, 0:1]
        l0.append((f > 0).sum(dim=1))
        l0_cut.append((f >= CUTOFF * f_max).sum(dim=1))
        multi = top2[:, 1] > 0
        second.append(top2[multi, 1] / top2[multi, 0])
        ratio = f / f_max
        secondary.append(ratio[(f > 0) & (ratio < 1)])
    second, secondary = torch.cat(second), torch.cat(secondary)
    return {"all": shares(torch.cat(l0)), "cut": shares(torch.cat(l0_cut)),
            "n_multi": len(second),
            "second_median": float(second.median()) if len(second) else float("nan"),
            "second_below_cutoff": float((second < CUTOFF).float().mean()) if len(second) else float("nan"),
            "secondary_median": float(secondary.median()) if len(secondary) else float("nan")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args(argv)
    torch.set_grad_enabled(False)
    results = {}
    for label, run, lam, subset in RUNS:
        path = Path(args.runs_dir) / run / "model.pt"
        if not path.exists():
            print(f"{label}: {path} not found, skipped")
            continue
        E = torch.as_tensor(data.load_prepared(args.data_dir, subset=subset)[0], dtype=torch.float32)
        model = ReLUSAE(E.shape[1], E.shape[0], lam)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        r = results[label] = spread(model, E)
        a, c = r["all"], r["cut"]
        print(f"{label}:")
        print(f"  active features per token: mean {a['mean']:.2f}, median {a['median']:g}, "
              f"max {a['max']}; share using 1 / 2 / 3 / 4+: {100 * a['1']:.1f}% / "
              f"{100 * a['2']:.1f}% / {100 * a['3']:.1f}% / {100 * a['4+']:.1f}%")
        print(f"  with the {CUTOFF:.0%} cutoff: mean {c['mean']:.2f}; share using 1 / 4+: "
              f"{100 * c['1']:.1f}% / {100 * c['4+']:.1f}%")
        print(f"  tokens using 2 or more: {r['n_multi']:,}; second coefficient / largest: median "
              f"{r['second_median']:.2f}, below {CUTOFF:.0%} for {100 * r['second_below_cutoff']:.0f}%; "
              f"all secondary coefficients / largest: median {r['secondary_median']:.2f}")
    return results


if __name__ == "__main__":
    main()
