# Experiment 4: grid of L1 ReLU SAEs on the subvocabulary

Status: script written (2026-10-21), not yet run. Runs on the laptop.

## Motivation

Part 1 of the write-up (`docs/write_up_plan.md`) documents failure modes on the 2,000-token subvocabulary. For the ReLU SAE it so far has only a sweep of the L1 coefficient. This experiment maps recovery over the hyperparameters that are free in the ReLU setting, to show how sensitive recovery is to them.

## Which hyperparameters vary

Of the 25 hyperparameters of the ReLU SAE, four vary and the rest stay fixed at the values used in Experiment 2:

| Hyperparameter | Values |
|---|---|
| L1 coefficient $\lambda$ | 0.03, 0.1, 0.2, 0.3, 0.5, 1 |
| Learning rate | $\tfrac{1}{4}$, 1 and 4 times the Gao et al. rule, about $1.4\times10^{-4}$, $5.7\times10^{-4}$ and $2.3\times10^{-3}$ at $m = 2{,}000$ |
| Batch size | 256, 1,024, 4,096 |
| Training length | Measured along each run at 0.5, 1, 2, 4 and 8 million tokens |

That is 6 × 3 × 3 = 54 cells, with one seed. Only large differences between cells will be interpreted, not differences between neighbouring cells.

The fixed settings are: $m = V$, unit-norm decoder rows, no decoder bias, a learned encoder bias initialised at zero, untied weights, ReLU activation, a constant $\lambda$, the L1 penalty $\sum_j f_j$, squared reconstruction error, a random Gaussian decoder with the encoder copied from it, seed 0, a constant learning rate, Adam with default settings, gradient projection and renormalisation for the decoder constraint, no clipping or weight decay, the 2,000 most frequent tokens sampled uniformly, and raw embeddings as input.

## Training length as a measurement axis

The learning rate is constant, so nothing in a run depends on its planned length. A run to 8 million tokens passes through the same states that shorter runs would. Each cell is therefore run once, and recovery is measured every half million tokens. Training length is counted in tokens, not steps, so cells with different batch sizes are compared at the same amount of data. A "million" is $2^{20}$ tokens, so that the measurement points fall on whole steps at every batch size. The 4-million point (4.2 million tokens) is close to the earlier $\lambda$ sweep's 4.1 million.

## Measurement

At every evaluation, over the whole subvocabulary: alive features, active features per token, relative loss, and decoder recovery at cosine above 0.9 and above 0.99 (mutual nearest neighbours). `summary.csv` keeps the rows at 0.5, 1, 2, 4 and 8 million tokens.

## How to run

```
caffeinate -i python experiments/exp4_relu_grid/run.py
```

The script times a few steps at each batch size and prints the expected time for the whole grid before it starts. It then trains the cells in turn. Results go to `runs/exp4/`: one folder per cell, with `eval_log.csv`, `result.json` and the final `model.pt` (about 12 MB), and `summary.csv` for the whole grid. A cell's checkpoint and initial parameters are deleted once it finishes.

Ctrl-C saves and stops. Running the same command again skips the finished cells and resumes the interrupted one. The heartbeat every minute shows the current cell and the ETA for the whole grid.

**Expected cost.** In the sandbox a cell took 11 to 18 minutes, depending on batch size: about 13 hours for the grid. The laptop ran Experiment 2's steps about 1.4 times faster, which suggests 9 to 10 hours there. The script prints its own estimate at the start. `--max-tokens-m 4` halves the time if needed, at the cost of the 8-million point.

## Code

`experiments/exp4_relu_grid/run.py` lists the grid values and calls `run_grid` in `sae_raw/grid.py`. Each cell is trained with the resumable `train()` in `sae_raw/train.py`, with decoder recovery added to every evaluation (`recovery_fields` in `sae_raw/evaluation.py`).
