"""Experiment 8: TopK annealing at intermediate vocabulary sizes
(plan: docs/experiment_8_topk_sizes.md)

One command runs everything:

    caffeinate -i python experiments/exp8_topk_sizes/run.py

For each size V in 5,000, 10,000 and 20,000, it trains a TopK SAE with m = V
on the V most frequent tokens. k is annealed with the shape of row T10 of the
write-up plan, starting from the k0 that leaves 93% of features alive at
initialisation (Experiment 5): 42, 55 and 71. The first 40% of training is
shared equally by k0 and the powers of 2 below it down to 4, then k = 2 for
40% and k = 1 for the last 20%. Batch size 512, the Gao learning rate, 4,000
steps, seed 0, 40 evaluations. Each evaluation also records n_top1, the
number of features that would survive a drop to k = 1 at that point.

--sizes runs only some of the sizes, for example --sizes 5000 10000.

Results go to runs/exp8/V<size>/: one folder per run (eval_log.csv,
result.json and the final model.pt) and summary.csv, with one row per
evaluation. Ctrl-C saves and stops; running the same command again resumes
where it left off, losing at most 10 minutes. A heartbeat every minute shows
progress and the ETA for all the chosen sizes. Everything is logged to
runs/exp8/V<size>/run.log.
"""

from sae_raw.evaluation import n_top1, recovery_fields
from sae_raw.grid import run_grids

K0_BY_SIZE = {5000: 42, 10000: 55, 20000: 71}  # 93% alive at initialisation (Experiment 5)
STEPS = 4000
BATCH_SIZE = 512
N_EVALS = 40


def matched_schedule(k0):
    """The T10 schedule shape, starting at k0.

    k0 and the powers of 2 below it, down to 4, share the first 40% of training
    equally; then k = 2 for 40% and k = 1 for the last 20%.
    """
    ks = [k0] + [2 ** i for i in range(k0.bit_length() - 1, 1, -1) if 2 ** i < k0]
    return {"shape": "stages", "stages": [[k, 0.4 / len(ks)] for k in ks] + [[2, 0.4], [1, 0.2]]}


def cells_for(V, k0, steps=STEPS, batch_size=BATCH_SIZE, n_evals=N_EVALS):
    return [{"name": f"V{V}_k0_{k0}", "model": "topk", "schedule": matched_schedule(k0),
             "steps": steps, "eval_every": steps // n_evals, "lr_factor": 1.0,
             "batch_size": batch_size}]


def eval_extra(model, E):
    return {**recovery_fields(model, E), "n_top1": n_top1(model, E)}


def main(argv=None):
    """Run the experiment; returns None if it was interrupted."""
    return run_grids("experiment_8_topk_sizes", __doc__, "runs/exp8",
                     {V: cells_for(V, k0) for V, k0 in K0_BY_SIZE.items()}, argv=argv,
                     eval_extra=eval_extra)


if __name__ == "__main__":
    main()
