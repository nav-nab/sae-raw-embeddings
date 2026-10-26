"""Experiment 7: TopK annealing on the subvocabulary, varying training length and learning rate
(plan: docs/experiment_7_topk_lr_steps.md)

One command runs everything:

    caffeinate -i python experiments/exp7_topk_lr_steps/run.py

Every cell trains a TopK SAE with m = V on the 2,000 most frequent tokens, with
k annealed by the schedule of row T10 of the write-up plan: k = 128, 64, 32, 16,
8 and 4, sharing the first 40% of training equally, then k = 2 for 40% and
k = 1 for the last 20%. Batch size 512, seed 0. The cells vary the total number
of steps and the learning rate (as a multiple of the Gao rule):

    steps 4,000 at 1x (the T10 baseline), 8,000 at 1x, 16,000 at 1x,
    steps 4,000 at 0.5x and at 2x.

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

import torch

from sae_raw.evaluation import recovery_fields
from sae_raw.grid import run_grid

SCHEDULE = {"shape": "stages",
            "stages": [[k, 0.4 / 6] for k in (128, 64, 32, 16, 8, 4)] + [[2, 0.4], [1, 0.2]]}
BATCH_SIZE = 512
N_EVALS = 40
CELLS_SPEC = [(4000, 1.0), (8000, 1.0), (16000, 1.0), (4000, 0.5), (4000, 2.0)]  # (steps, lr factor)


def make_topk_cells(spec, schedule=SCHEDULE, batch_size=BATCH_SIZE, n_evals=N_EVALS):
    cells = []
    for steps, lrf in spec:
        if steps % n_evals:
            raise ValueError(f"{steps} steps do not divide into {n_evals} evaluations")
        cells.append({"name": f"steps{steps}_lr{lrf:g}", "model": "topk", "schedule": schedule,
                      "steps": steps, "eval_every": steps // n_evals, "lr_factor": lrf,
                      "batch_size": batch_size})
    return cells


@torch.no_grad()
def n_top1(model, E, chunk=1024):
    """Features that are the top feature, with a positive pre-activation, for some token."""
    winners = torch.zeros(model.m, dtype=torch.bool, device=E.device)
    for start in range(0, E.shape[0], chunk):
        a = E[start:start + chunk] @ model.W_enc.T + model.b_enc
        val, idx = a.max(dim=1)
        winners[idx[val > 0]] = True
    return int(winners.sum())


def eval_extra(model, E):
    return {**recovery_fields(model, E), "n_top1": n_top1(model, E)}


if __name__ == "__main__":
    run_grid("experiment_7_topk_lr_steps", __doc__, "runs/exp7", make_topk_cells(CELLS_SPEC),
             eval_extra=eval_extra)
