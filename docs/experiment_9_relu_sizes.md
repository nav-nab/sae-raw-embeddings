# Experiment 9: L1 ReLU SAEs at intermediate vocabulary sizes

Status: script written and checked, 2026-10-26. To be run on the laptop over two nights.

## Question

Part 3b of `docs/write_up_plan.md`: how does the training needed for a given $\lambda$ grow with the vocabulary size $V$, and does the best $\lambda$ change? On the subset, every $\lambda$ from 0.03 to 0.5 recovered over 80% given enough training, and the smallest $\lambda$ recovered most but needed longest (Experiment 4). At full scale, only $\lambda = 0.3$ has been run (Experiment 2: 93% after 85,100 steps).

The geometry of the embeddings barely changes with $V$ (Experiment 6), so a change in recovery with $V$ can be put down to training.

## Design

- **Vocabularies:** the 5,000, 10,000 and 20,000 most frequent tokens, as in Experiment 8. $m = V$.
- **$\lambda$:** 0.1, 0.3 and 1 at every size, and 0.03 at 5,000 tokens only. At the larger sizes, $\lambda = 0.03$ would need 1 to 3 days per run.
- **Training:** batch size 256 at the Gao learning rate, the best region of Experiment 4. Seed 0.
- **Planned length:** in proportion to $V$, from a rough interpolation between the subset ($\lambda = 0.3$ converged in about 4,000 steps) and Experiment 2 (about 85,000 steps at full scale):

| $\lambda$ | Planned steps | At $V$ = 5,000 | 10,000 | 20,000 |
|---|---|---|---|---|
| 1 | $V$ | 5,000 | 10,000 | 20,000 |
| 0.3 | $2V$ | 10,000 | 20,000 | 40,000 |
| 0.1 | $4V$ | 20,000 | 40,000 | 80,000 |
| 0.03 | $16V$ | 80,000 | | |

- **Measurement:** 40 evaluations per run, with decoder recovery, alive features and the spread of active features per token (median, maximum, and the shares using 1, 2, 3 and 4 or more). The learning rate is constant, so the evaluations along a run show every shorter training length too.
- **Extension:** every run keeps its checkpoint and initial parameters when it finishes, so that a run that has not converged can be trained further with `--extend`.

## Expected cost

| Size | Runs | Sandbox estimate | Disk kept |
|---|---|---|---|
| 5,000 | 4 | 2.0 h | 0.6 GB |
| 10,000 | 3 | 3.2 h | 0.9 GB |
| 20,000 | 3 | 17 to 19 h | 1.8 GB |

The laptop ran Experiment 2's steps about 1.4 times faster than the sandbox, so night 1 (5,000 and 10,000) should take about 4 hours and night 2 (20,000) about 12 to 14 hours.

## How to run

```
caffeinate -i python experiments/exp9_relu_sizes/run.py --sizes 5000 10000   # night 1
caffeinate -i python experiments/exp9_relu_sizes/run.py --sizes 20000        # night 2
```

To train a run further, give its new total number of steps:

```
caffeinate -i python experiments/exp9_relu_sizes/run.py --sizes 10000 --extend V10000_lam0.1 60000
```

`--extend` sets the total, so giving the same command twice does not extend the run twice. After an interrupt, the same command (with or without `--extend`) resumes. Several `--extend` options can be given at once.

Results go to `runs/exp9/V<size>/`: one folder per run (`eval_log.csv`, `result.json`, `model.pt`, and `checkpoint.pt` and `init.pt` for resuming) and `summary.csv`, with one row per evaluation. Ctrl-C saves and stops, and the same command resumes, losing at most 10 minutes. A heartbeat every minute shows the ETA for all the chosen sizes.

## Code

`experiments/exp9_relu_sizes/run.py` defines the runs and calls `run_grids` in `sae_raw/grid.py`. `code_spread_fields` in `sae_raw/evaluation.py` measures the spread of active features per token.
