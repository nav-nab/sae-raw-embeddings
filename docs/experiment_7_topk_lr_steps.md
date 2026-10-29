# Experiment 7: TopK annealing with more training and other learning rates

Status: done. The first five cells ran on the laptop on 2026-10-26 to 27, and an extension with extra seeds and longer runs on 2026-10-28 to 29. The extension's results (at the end) revise finding 2 of the first five cells.

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
2. ~~**Each doubling gains less.**~~ **Revised by the extension.** Per doubling of learning rate × steps, recovery here rose by 69, then about 47, then about 26 tokens, which suggested a plateau near 80%. The extension's seeds and longer runs show the gains do not shrink: this apparent pattern was noise in single runs.
3. **More training helps in two ways.** Fewer features die at the drop from $k = 4$ to $k = 2$: 1,935 are alive at the end of the $k = 2$ stage, against 1,883. More features are the top feature of some token at the end of the $k = 2$ stage: 1,615 against 1,541. These are the features that survive the drop to $k = 1$.
4. **`n_top1` rises during the $k = 2$ stage, then levels off.** In the baseline it rises from 1,395 to 1,541, mostly in the first quarter of the stage, while the alive count stays flat at about 1,880. This explains why a longer $k = 2$ stage did not help in the Experiment 3 grid, while more training overall does.
5. **The drop to $k = 1$ remains the bottleneck.** It costs 18% to 20% of the alive features in every cell. Recovered tokens are 96% to 100% of the alive features.

In the 4,000-step cell at 2 times the rate, recovered tokens (1,555) slightly exceed alive features (1,553). A feature that has just lost its last token at the drop can keep a decoder row that still matches that token.

## Extension (2026-10-28 to 29)

Two questions remained: are the gains from more training real, and do they keep shrinking? The extension added seven cells to the same grid: seeds 1 and 2 at 8,000 and 16,000 steps (three seeds in all at each length), and single runs at 32,000, 64,000 and 128,000 steps, all at the Gao rate. About 6 hours on the laptop.

**Seeds.**

| Steps | Seed 0 | Seed 1 | Seed 2 | Mean | Range |
|---|---|---|---|---|---|
| 8,000 | 1,529 | 1,555 | 1,515 | 1,533 | 40 |
| 16,000 | 1,568 | 1,578 | 1,606 | 1,584 | 38 |

**Longer runs.**

| Steps | Epochs' worth | Alive at end of $k = 2$ | `n_top1` at end of $k = 2$ | Alive at end | Recovered > 0.9 | Recovered > 0.99 | Lost at the drop to $k = 1$ | Relative loss |
|---|---|---|---|---|---|---|---|---|
| 4,000 | 1,024 | 1,883 | 1,541 | 1,515 | 1,495 (75%) | 1,241 | 20% | 0.062 |
| 8,000 (mean of 3) | 2,048 | 1,928 | 1,585 | 1,547 | 1,533 (77%) | 1,272 | 20% | 0.057 |
| 16,000 (mean of 3) | 4,096 | 1,935 | 1,633 | 1,595 | 1,584 (79%) | 1,341 | 18% | 0.049 |
| 32,000 | 8,192 | 1,957 | 1,689 | 1,677 | 1,668 (83%) | 1,492 | 14% | 0.036 |
| 64,000 | 16,384 | 1,953 | 1,717 | 1,709 | 1,706 (85%) | 1,553 | 13% | 0.030 |
| 128,000 | 32,768 | 1,948 | 1,784 | 1,771 | **1,771 (89%)** | 1,644 | 9% | 0.023 |

### Findings of the extension

1. **The gains from more training are real.** The three-seed means at 8,000 and 16,000 steps differ by 51 tokens, and the ranges do not overlap: the worst seed at 16,000 steps beats the best at 8,000. The spread between seeds, about 40 tokens, matches the difference between machines found earlier, so a single run is good to about ±20 tokens.
2. **The gains do not shrink.** Per doubling, recovery rose by about 38, 51, 84, 38 and 65 tokens, roughly 3 percentage points each time, with no sign of levelling off by 128,000 steps. Both predictions made before the run were too low (about 1,590 if the gains kept halving, about 1,670 if they held at 30 to 40 per doubling).
3. **More training now helps only through the drop to $k = 1$.** From about 32,000 steps on, about 1,950 features are alive at the end of the $k = 2$ stage in every run, so more training no longer saves features before then. The further gains come from more features being some token's top feature at the drop (`n_top1`: 1,541 at 4,000 steps, 1,784 at 128,000), so the drop costs 9% of the alive features instead of 20%.
4. **The $k = 1$ stage changes nothing.** At 128,000 steps, the alive count and recovery stay flat throughout it (1,770 to 1,771), and every alive feature is a recovered token.
5. **Precision improves fastest.** Recovery above 0.99 rose from 62% to 82%.
6. **TopK can approach ReLU's recovery, but at a much higher cost.** The best ReLU run on the same subset recovered 98% in 32,768 steps at batch size 256 (about 8 million training tokens). The 128,000-step TopK run recovered 89% with about 66 million training tokens.

We stopped at 128,000 steps. Since the gains were not shrinking, it is a fair conjecture that longer runs would recover more than 89%.
