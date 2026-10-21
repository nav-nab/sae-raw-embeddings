"""Experiment 4: grid of L1 ReLU SAEs on the subvocabulary (plan: docs/experiment_4_relu_grid.md)

One command runs everything:

    caffeinate -i python experiments/exp4_relu_grid/run.py

It trains an L1 ReLU SAE with m = V on the 2,000 most frequent tokens for every
combination of the L1 coefficient, the learning rate (as a multiple of the Gao
et al. rule) and the batch size below: 54 cells, one seed. Each cell trains to
8 million tokens (2^23), and decoder recovery is measured every half million,
so the effect of training length is read off along each run. All other
settings are as in Experiment 2.

Results go to runs/exp4/: one folder per cell (eval_log.csv, result.json and
the final model.pt) and summary.csv with one row per cell and token count.
Ctrl-C saves and stops; running the same command again skips finished cells
and resumes the interrupted one. A heartbeat every minute shows progress and
the ETA for the whole grid. Everything is logged to runs/exp4/run.log.
"""

from sae_raw.grid import make_cells, run_grid

L1_COEFS = [0.03, 0.1, 0.2, 0.3, 0.5, 1.0]
LR_FACTORS = [0.25, 1.0, 4.0]  # multiples of 2e-4 / sqrt(m / 2^14)
BATCH_SIZES = [256, 1024, 4096]

if __name__ == "__main__":
    run_grid("experiment_4_relu_grid", __doc__, "runs/exp4",
             make_cells(L1_COEFS, LR_FACTORS, BATCH_SIZES))
