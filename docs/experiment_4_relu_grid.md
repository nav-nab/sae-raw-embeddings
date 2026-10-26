# Experiment 4: grid of L1 ReLU SAEs on the subvocabulary

Status: done. Run on the laptop on 2026-10-21 to 22, 9.5 hours for the 54 cells. Results below.

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

## Results

All counts are out of 2,000. "Recovered" counts tokens with a mutual-nearest decoder row at cosine above the threshold.

**Recovered at cosine > 0.9, at 8 million tokens.** Columns are the learning rate (as a multiple of the Gao rule) and the batch size.

| $\lambda$ | ¼, 256 | ¼, 1,024 | ¼, 4,096 | 1, 256 | 1, 1,024 | 1, 4,096 | 4, 256 | 4, 1,024 | 4, 4,096 |
|---|---|---|---|---|---|---|---|---|---|
| 0.03 | 330 | 1 | 0 | **1,959** | 555 | 1 | 1,372 | 1,065 | 321 |
| 0.1 | 1,881 | 440 | 3 | 1,861 | 1,899 | 75 | 1,128 | 1,190 | 827 |
| 0.2 | 1,793 | 1,632 | 17 | 1,794 | 1,788 | 945 | 760 | 762 | 748 |
| 0.3 | 1,724 | 1,705 | 77 | 1,753 | 1,751 | 1,315 | 430 | 384 | 399 |
| 0.5 | 1,667 | 1,631 | 302 | 1,640 | 1,612 | 1,438 | 51 | 27 | 35 |
| 1 | 1,346 | 1,296 | 428 | 881 | 709 | 626 | 0 | 0 | 0 |

**Recovered at cosine > 0.99, at 8 million tokens.**

| $\lambda$ | ¼, 256 | ¼, 1,024 | ¼, 4,096 | 1, 256 | 1, 1,024 | 1, 4,096 | 4, 256 | 4, 1,024 | 4, 4,096 |
|---|---|---|---|---|---|---|---|---|---|
| 0.03 | 186 | 0 | 0 | **1,883** | 329 | 1 | 1,050 | 839 | 238 |
| 0.1 | 1,812 | 164 | 0 | 1,786 | 1,836 | 10 | 783 | 899 | 562 |
| 0.2 | 1,697 | 1,298 | 1 | 1,676 | 1,687 | 489 | 411 | 413 | 415 |
| 0.3 | 1,594 | 1,484 | 1 | 1,619 | 1,626 | 820 | 165 | 105 | 124 |
| 0.5 | 1,509 | 1,423 | 50 | 1,470 | 1,431 | 1,016 | 4 | 4 | 3 |
| 1 | 1,057 | 989 | 62 | 567 | 426 | 332 | 0 | 0 | 0 |

**Alive features at 8 million tokens.**

| $\lambda$ | ¼, 256 | ¼, 1,024 | ¼, 4,096 | 1, 256 | 1, 1,024 | 1, 4,096 | 4, 256 | 4, 1,024 | 4, 4,096 |
|---|---|---|---|---|---|---|---|---|---|
| 0.03 | 2,000 | 2,000 | 2,000 | 2,000 | 2,000 | 2,000 | 1,518 | 1,688 | 1,685 |
| 0.1 | 1,970 | 2,000 | 2,000 | 1,959 | 1,958 | 2,000 | 1,300 | 1,433 | 1,434 |
| 0.2 | 1,915 | 1,947 | 2,000 | 1,902 | 1,903 | 1,995 | 972 | 1,027 | 1,041 |
| 0.3 | 1,871 | 1,924 | 2,000 | 1,882 | 1,875 | 1,959 | 686 | 657 | 682 |
| 0.5 | 1,873 | 1,877 | 1,995 | 1,805 | 1,751 | 1,770 | 239 | 209 | 220 |
| 1 | 1,600 | 1,510 | 1,601 | 1,048 | 895 | 814 | 7 | 3 | 4 |

The cell matching the earlier $\lambda$ sweep ($\lambda = 0.3$, Gao rate, batch 1,024) recovered 1,717 tokens at 4 million tokens, against 1,710 in the sweep.

### Findings

**1. Two failure modes, on opposite sides of the working region.**

- **Too little training: codes stay dense.** Every feature stays alive, but tokens use many features, so few are recovered. This is the failure at a small $\lambda$, a low learning rate or a large batch. For example, at $\lambda = 0.03$ with batch size 4,096, tokens still use 175 to 378 active features after 8 million tokens.
- **Too much pressure: features die.** Recovery is then capped by the number of alive features. Deaths rise with both $\lambda$ and the learning rate. At 4 times the Gao rate, $\lambda = 1$ leaves 3 to 7 features alive. These deaths happen in the first half-million tokens: at 4 times the Gao rate, the alive count barely changes after the first measurement. They do not depend on the batch size.

**2. Sparsification runs on learning rate × steps, not on tokens seen.** Measure training in Gao-rate steps: the number of steps times the learning rate as a multiple of the Gao rule. Below the death regime, cells with the same number of Gao-rate steps reach similar sparsity, whatever their batch size. At $\lambda = 0.1$, for example, three cells at 128 Gao-rate steps all have about 330 active features per token. They differ in batch size (1,024 and 4,096) and learning rate (¼ and 1). Adam moves each weight by about the learning rate per step, whatever the batch size, so a larger batch buys a less noisy gradient but no more progress. At a fixed number of tokens, a smaller batch therefore means more steps and more progress.

**3. A weaker penalty recovers more, but needs far more training.** Once codes have sparsified, the best recovery falls steadily with $\lambda$: 1,959 at 0.03, about 1,890 at 0.1, about 1,790 at 0.2, about 1,750 at 0.3, about 1,650 at 0.5 and about 1,350 at 1. A larger $\lambda$ kills more features on the way. A smaller $\lambda$ sparsifies more slowly: $\lambda = 0.3$ is done by about 4,096 Gao-rate steps, $\lambda = 0.1$ by about 8,192, and $\lambda = 0.03$ only in the longest cell (32,768 steps at batch size 256). The best result of the grid, 1,959 tokens (98%) at cosine above 0.9 and 1,883 above 0.99, is at $\lambda = 0.03$.

This revises the reading of the earlier $\lambda$ sweep. That sweep ran every $\lambda$ for the same 4,000 steps. It found $\lambda = 0.03$ "too weak", but the codes there were simply not yet sparse: given enough steps, $\lambda = 0.03$ does best.
