# Experiment 7: TopK annealing with more training and other learning rates

Status: script written and checked, 2026-10-26. To be run on the laptop.

## Motivation

The best TopK result on the subset is T10 in `docs/write_up_plan.md`: 1,499 of 2,000 tokens recovered at cosine above 0.9 (75%). Nearly every surviving feature is correct (1,499 recovered of 1,523 alive), so the whole shortfall is features dying.

The deaths happen at the drops in $k$, and mostly at the drop from $k = 2$ to $k = 1$. In T10, the alive count was 1,989 at the end of the $k = 4$ stage, 1,882 to 1,886 throughout the $k = 2$ stage, and 1,523 throughout the $k = 1$ stage. At the moment of the drop, a feature survives only if it is the top feature of at least one token. A feature that is second for all its tokens dies at once and never revives.

The subset grid of Experiment 3 has ruled out the schedule as a lever. Recovery saturates in the starting $k$, and the length of the $k = 2$ stage makes no difference: the alive count stays flat through it. But every TopK subset run used the same learning rate (the Gao rule), batch size (512) and length (4,000 steps). In the ReLU grid (Experiment 4), learning rate × steps was the lever that mattered most. This experiment tests whether it matters for TopK too.

The expected gain is modest. The flat alive count during the $k = 2$ stage suggests that the losses come from the structure of 2-sparse codes rather than from undertraining. A negative result would still strengthen the TopK conclusions.

## Design

- **Data and model:** as T10. The 2,000 most frequent tokens, a TopK SAE with $m = 2{,}000$, seed 0, batch size 512.
- **Schedule:** as T10. $k$ = 128, 64, 32, 16, 8 and 4 share the first 40% of training equally, then $k = 2$ for 40% and $k = 1$ for the last 20%. The schedule scales with the total number of steps.
- **Cells:**

| Cell | Steps | Learning rate (× the Gao rule) |
|---|---|---|
| `steps4000_lr1` | 4,000 | 1 (the T10 baseline) |
| `steps8000_lr1` | 8,000 | 1 |
| `steps16000_lr1` | 16,000 | 1 |
| `steps4000_lr0.5` | 4,000 | ½ |
| `steps4000_lr2` | 4,000 | 2 |

- **Measurement:** 40 evaluations per cell, so evaluations fall at the end of the $k = 4$ stage (40%) and of the $k = 2$ stage (80%). Each records the usual fields and decoder recovery. It also records `n_top1`: the number of features that are the top feature, with a positive pre-activation, for at least one token. At the end of the $k = 2$ stage, `n_top1` is the number of features that can survive the drop to $k = 1$.

## Check of the baseline (sandbox, 2026-10-26)

The baseline cell was run once in the sandbox, with the grid code that the laptop run will use. It recovered 1,531 tokens at cosine above 0.9 and 1,292 above 0.99, with 1,552 features alive. The original T10 pilot, run with an earlier script, gave 1,499, 1,250 and 1,523. The two runs share the seed and settings but not the code path, so they differ in detail. A difference of about 30 tokens is therefore within what reimplementation alone can produce. The write-up should compare the cells of this experiment with its own baseline, not with T10.

At the end of the $k = 2$ stage, the baseline had `n_top1` = 1,571 against 1,877 alive. After the drop, 1,552 were alive. So `n_top1` predicts the survivors to within about 1%.

## How to run

```
caffeinate -i python experiments/exp7_topk_lr_steps/run.py
```

Results go to `runs/exp7/`: one folder per cell (`eval_log.csv`, `result.json` and the final `model.pt`) and `summary.csv`, with one row per cell and evaluation. Ctrl-C saves and stops, and the same command resumes. A heartbeat every minute shows the ETA for the whole grid.

**Expected cost.** The script's estimate in the sandbox was 1 hour 34 minutes for the five cells, about half of it in the 16,000-step cell. Steps at $k = 128$ take most of the time.

## Code

`experiments/exp7_topk_lr_steps/run.py` defines the cells and the `n_top1` measurement, and calls `run_grid` in `sae_raw/grid.py`. The grid runner now takes TopK cells as well as L1 cells: a TopK cell has its own schedule, number of steps and evaluation interval.
