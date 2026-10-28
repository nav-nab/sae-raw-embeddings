"""Experiment 7: TopK annealing on the subvocabulary, varying training length and learning rate
(plan: docs/experiment_7_topk_lr_steps.md)

One command runs everything:

    caffeinate -i python experiments/exp7_topk_lr_steps/run.py

Every cell trains a TopK SAE with m = V on the 2,000 most frequent tokens, with
k annealed by the schedule of row T10 of the write-up plan: k = 128, 64, 32, 16,
8 and 4, sharing the first 40% of training equally, then k = 2 for 40% and
k = 1 for the last 20%. Batch size 512. The cells vary the total number of
steps, the learning rate (as a multiple of the Gao rule) and the seed:

    original grid (seed 0): 4,000 steps at 1x (the T10 baseline), 8,000 and
        16,000 at 1x, 4,000 at 0.5x and at 2x;
    extension (2026-10-28): 8,000 and 16,000 steps at 1x with seeds 1 and 2,
        and 32,000, 64,000 and 128,000 steps at 1x with seed 0.

Running the script again after the original grid has finished adds the
extension cells to the same grid and runs only those.

Each cell is evaluated 40 times, so evaluations fall at the end of the k = 4
stage (40%) and of the k = 2 stage (80%). Besides the usual fields, each
evaluation records n_top1: the number of features that are the top feature,
with a positive pre-activation, for at least one token. At the end of the
k = 2 stage this is the number of features that survive the drop to k = 1.

Results go to runs/exp7/: one folder per cell (eval_log.csv, result.json and
the final model.pt) and summary.csv with one row per cell and evaluation.
Ctrl-C saves and stops; running the same command again skips finished cells
and resumes the interrupted one. A heartbeat every minute shows progress and
the ETA for the whole grid. Everything is logged to runs/exp7/run.log.
"""

from sae_raw.evaluation import n_top1, recovery_fields
from sae_raw.grid import run_grid

SCHEDULE = {"shape": "stages",
            "stages": [[k, 0.4 / 6] for k in (128, 64, 32, 16, 8, 4)] + [[2, 0.4], [1, 0.2]]}
BATCH_SIZE = 512
N_EVALS = 40
# (steps, lr factor, seed). The first five are the original grid; the rest are the extension
# (2026-10-28): two more seeds at 8,000 and 16,000 steps, and longer runs.
CELLS_SPEC = [(4000, 1.0, 0), (8000, 1.0, 0), (16000, 1.0, 0), (4000, 0.5, 0), (4000, 2.0, 0),
              (8000, 1.0, 1), (8000, 1.0, 2), (16000, 1.0, 1), (16000, 1.0, 2),
              (32000, 1.0, 0), (64000, 1.0, 0), (128000, 1.0, 0)]


def make_topk_cells(spec, schedule=SCHEDULE, batch_size=BATCH_SIZE, n_evals=N_EVALS):
    cells = []
    for steps, lrf, *seed in spec:
        seed = seed[0] if seed else 0
        if steps % n_evals:
            raise ValueError(f"{steps} steps do not divide into {n_evals} evaluations")
        name = f"steps{steps}_lr{lrf:g}" + (f"_seed{seed}" if seed else "")
        cells.append({"name": name, "model": "topk", "schedule": schedule, "steps": steps,
                      "eval_every": steps // n_evals, "lr_factor": lrf,
                      "batch_size": batch_size, "seed": seed})
    return cells


def eval_extra(model, E):
    return {**recovery_fields(model, E), "n_top1": n_top1(model, E)}


def main(argv=None):
    """Run the experiment; returns None if it was interrupted."""
    return run_grid("experiment_7_topk_lr_steps", __doc__, "runs/exp7",
                    make_topk_cells(CELLS_SPEC), argv=argv, eval_extra=eval_extra)


if __name__ == "__main__":
    main()
