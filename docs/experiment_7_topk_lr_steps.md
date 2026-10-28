# Experiment 7: TopK annealing with more training and other learning rates

Status: first five cells run on the laptop on 2026-10-26 to 27 (night 1). Results below. An extension with extra seeds and longer runs is set up; see the end.

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

## Results (laptop, 2026-10-27)

| Cell | Gao-rate steps (learning rate × steps) | Alive at end of $k = 2$ | `n_top1` at end of $k = 2$ | Alive at end | Recovered > 0.9 | Recovered > 0.99 |
|---|---|---|---|---|---|---|
| 4,000 steps at ½ | 2,000 | 1,783 | 1,497 | 1,479 | 1,426 (71%) | 1,187 |
| 4,000 steps at 1 (baseline) | 4,000 | 1,883 | 1,541 | 1,515 | 1,495 (75%) | 1,241 |
| 8,000 steps at 1 | 8,000 | 1,935 | 1,578 | 1,537 | 1,529 (76%) | 1,266 |
| 4,000 steps at 2 | 8,000 | 1,923 | 1,593 | 1,553 | 1,555 (78%) | 1,300 |
| 16,000 steps at 1 | 16,000 | 1,935 | 1,615 | 1,589 | 1,568 (78%) | 1,333 |

**Noise.** The baseline cell gave 1,495 tokens on the laptop and 1,531 in the sandbox, with the same code, seed and settings. Differences of about 35 tokens between single runs can therefore come from the machine alone. On that scale, the 8,000-step gain (+34) is within the noise, while the others (−69 at ½, +60 at 2× and +73 at 16,000 steps) are beyond it. All five cells fall in order of learning rate × steps.

### Findings

1. **TopK follows the same rule as ReLU: progress depends on learning rate × steps.** The two cells with 8,000 Gao-rate steps give similar results (1,529 and 1,555).
2. **Each doubling gains less.** Per doubling of learning rate × steps, recovery rose by 69, then about 47, then about 26 tokens. If the gains keep halving, recovery levels off near 1,595 (80%).
3. **More training helps in two ways.** Fewer features die at the drop from $k = 4$ to $k = 2$: 1,935 are alive at the end of the $k = 2$ stage, against 1,883. More features are the top feature of some token at the end of the $k = 2$ stage: 1,615 against 1,541. These are the features that survive the drop to $k = 1$.
4. **`n_top1` rises during the $k = 2$ stage, then levels off.** In the baseline it rises from 1,395 to 1,541, mostly in the first quarter of the stage, while the alive count stays flat at about 1,880. This explains why a longer $k = 2$ stage did not help in the Experiment 3 grid, while more training overall does.
5. **The drop to $k = 1$ remains the bottleneck.** It costs 18% to 20% of the alive features in every cell. Recovered tokens are 96% to 100% of the alive features.

In the 4,000-step cell at 2 times the rate, recovered tokens (1,555) slightly exceed alive features (1,553). A feature that has just lost its last token at the drop can keep a decoder row that still matches that token.

## Extension (set up 2026-10-28)

Two questions remain: are the gains from more training real, and do they keep shrinking? The extension adds seven cells to the same grid:

| Cells | Purpose |
|---|---|
| 8,000 and 16,000 steps at the Gao rate, seeds 1 and 2 | With seed 0, three seeds each, to measure the run-to-run spread |
| 32,000, 64,000 and 128,000 steps at the Gao rate, seed 0 | To see whether the gains keep halving with each doubling |

If the gains keep halving, the 128,000-step run should recover about 1,590 tokens. If they stay at about 30 to 40 per doubling, it should recover about 1,670. The difference is well above the noise.

**Expected cost on the laptop:** about 1 hour for the four seed cells and about 5 hours for the three long runs, about 6 hours in all. The same command runs only the new cells:

```
caffeinate -i python experiments/exp7_topk_lr_steps/run.py
```
