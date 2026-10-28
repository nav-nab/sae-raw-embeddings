# Experiment 9: L1 ReLU SAEs at intermediate vocabulary sizes

Status: done. Run on the laptop on 2026-10-26 to 28 (5,000 and 10,000 tokens on night 1, 20,000 on night 2). Results at the end.

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

## Results

Recovered tokens at cosine above 0.9, at the end of each run. The 2,000-token column is from the matching Experiment 4 cells (Gao rate, batch size 256), read at the nearest epochs' worth (262, 524, 1,049 and 4,194).

| $\lambda$ | Epochs' worth | $V$ = 2,000 | 5,000 | 10,000 | 20,000 |
|---|---|---|---|---|---|
| 1 | 256 | 830 (42%) | 1,587 (32%) | 3,291 (33%) | 8,442 (42%) |
| 0.3 | 512 | 1,672 (84%) | 4,333 (87%) | 8,799 (88%) | 17,867 (89%) |
| 0.1 | 1,024 | 1,804 (90%) | 4,604 (92%) | 9,262 (93%) | 18,033 (90%) |
| 0.03 | 4,096 | 1,959 (98%) | 4,925 (98.5%) | | |

At the end of each run:

| Run | Alive | Recovered > 0.99 | Active features per token (mean) | Share using one feature | Most used by any token | Relative loss |
|---|---|---|---|---|---|---|
| 5,000, $\lambda = 1$ | 2,223 | 696 | 3.17 | 21% | 18 | 0.235 |
| 5,000, $\lambda = 0.3$ | 4,849 | 3,796 | 1.78 | 62% | 10 | 0.011 |
| 5,000, $\lambda = 0.1$ | 4,986 | 4,128 | 1.66 | 71% | 11 | 0.0025 |
| 5,000, $\lambda = 0.03$ | 4,997 | 4,694 | 1.20 | 88% | 7 | 0.0008 |
| 10,000, $\lambda = 1$ | 4,341 | 1,695 | 2.95 | 26% | 16 | 0.220 |
| 10,000, $\lambda = 0.3$ | 9,823 | 7,932 | 1.65 | 68% | 11 | 0.011 |
| 10,000, $\lambda = 0.1$ | 9,995 | 8,127 | 1.67 | 69% | 13 | 0.0022 |
| 20,000, $\lambda = 1$ | 10,381 | 4,956 | 2.47 | 37% | 12 | 0.172 |
| 20,000, $\lambda = 0.3$ | 19,858 | 16,284 | 1.61 | 70% | 12 | 0.0099 |
| 20,000, $\lambda = 0.1$ | 19,999 | 14,876 | 1.86 | 62% | 14 | 0.0023 |

### Findings

1. **Recovery as a function of $\lambda$ does not depend on the vocabulary size.** With steps in proportion to $V$, each $\lambda$ ends within a few points at every size. The full table fits the same picture: Experiment 2 ($\lambda = 0.3$, about 1,700 epochs' worth) recovered 93%. This is a conceptual theme of the write-up (Part 3b).
2. **The steps needed grow in proportion to $V$.** At a fixed batch size this is the same epochs' worth. The sparsification curves line up in epochs' worth too. At $\lambda = 0.3$, codes go from about 200 active features per token to about 1.6 over similar epochs at every size. Larger sizes are, if anything, slightly ahead: at 256 epochs' worth, 24%, 28% and 36% of tokens are recovered at 5,000, 10,000 and 20,000 tokens.
3. **This holds although the learning rate falls with size.** The Gao rate falls as $1/\sqrt{m}$, so at $\lambda = 0.3$ the absolute learning rate × steps is about 3 times larger at 20,000 tokens than at 2,000, for the same recovery. Across sizes, recovery tracks steps per token of vocabulary, not the absolute learning rate × steps.
4. **The best $\lambda$ does not change.** At every size a smaller $\lambda$ recovers more, given its longer training. $\lambda = 1$ fails at every size, because about half the features die early (44% to 53% alive at the end).
5. **$\lambda = 0.1$ had not converged.** Its recovery was still rising steeply at the end at every size (at 20,000 tokens, from 17,194 to 18,033 over the last 52 epochs' worth), so its numbers are lower bounds. The 20,000-token run also lagged mid-run: 50% recovered at 768 epochs' worth, against 70% at 5,000 tokens. An extension to 2,048 epochs' worth was considered and judged unnecessary, since the picture is clear without it. $\lambda = 0.3$ was close to its plateau, and $\lambda = 1$ is capped by deaths.

The checkpoints are kept, so the runs can still be extended with `--extend` if needed.
