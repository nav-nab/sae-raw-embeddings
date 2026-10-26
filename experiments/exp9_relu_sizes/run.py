"""Experiment 9: L1 ReLU SAEs at intermediate vocabulary sizes
(plan: docs/experiment_9_relu_sizes.md)

One command runs everything:

    caffeinate -i python experiments/exp9_relu_sizes/run.py

For each size V in 5,000, 10,000 and 20,000, it trains L1 ReLU SAEs with
m = V on the V most frequent tokens, at lambda = 0.1, 0.3 and 1, and also at
lambda = 0.03 for V = 5,000. Batch size 256, the Gao learning rate, seed 0.
The planned lengths grow in proportion to V: 4V steps at lambda = 0.1, 2V at
0.3, V at 1 and 16V at 0.03. Each run is evaluated 40 times along the way,
with decoder recovery and the spread of active features per token. The
learning rate is constant, so the evaluations show every shorter length too.

--sizes runs only some of the sizes. The planned split over two nights is

    caffeinate -i python experiments/exp9_relu_sizes/run.py --sizes 5000 10000
    caffeinate -i python experiments/exp9_relu_sizes/run.py --sizes 20000

Every run keeps its checkpoint when it finishes, so a run that has not
converged can be trained further:

    caffeinate -i python experiments/exp9_relu_sizes/run.py --sizes 10000 --extend V10000_lam0.1 60000

--extend CELL STEPS sets the run's new total number of steps, so giving the
same command twice does not extend it twice.

Results go to runs/exp9/V<size>/: one folder per run (eval_log.csv,
result.json, model.pt, and checkpoint.pt and init.pt for resuming) and
summary.csv, with one row per evaluation. Ctrl-C saves and stops; running the
same command again resumes where it left off, losing at most 10 minutes. A
heartbeat every minute shows progress and the ETA for all the chosen sizes.
Everything is logged to runs/exp9/V<size>/run.log.
"""

from sae_raw.evaluation import code_spread_fields, recovery_fields
from sae_raw.grid import run_grids

SIZES = [5000, 10000, 20000]
STEPS_PER_TOKEN = {0.1: 4, 0.3: 2, 1.0: 1}  # planned steps = this many times V
EXTRA = {5000: {0.03: 16}}  # lambda = 0.03 only at the smallest size
BATCH_SIZE = 256
N_EVALS = 40


def cells_for(V, steps_per_token, batch_size=BATCH_SIZE, n_evals=N_EVALS):
    cells = []
    for lam, mult in sorted(steps_per_token.items(), reverse=True):  # fastest first
        steps = mult * V
        if steps % n_evals:
            raise ValueError(f"{steps} steps do not divide into {n_evals} evaluations")
        cells.append({"name": f"V{V}_lam{lam:g}", "l1_coef": lam, "steps": steps,
                      "eval_every": steps // n_evals, "lr_factor": 1.0,
                      "batch_size": batch_size, "keep_checkpoint": True})
    return cells


def all_cells():
    return {V: cells_for(V, {**STEPS_PER_TOKEN, **EXTRA.get(V, {})}) for V in SIZES}


def eval_extra(model, E):
    return {**recovery_fields(model, E), **code_spread_fields(model, E)}


def main(argv=None):
    """Run the experiment; returns None if it was interrupted."""
    return run_grids("experiment_9_relu_sizes", __doc__, "runs/exp9", all_cells(), argv=argv,
                     eval_extra=eval_extra)


if __name__ == "__main__":
    main()
