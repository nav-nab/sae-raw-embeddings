# Experiment 8: TopK annealing at intermediate vocabulary sizes

Status: script written and checked, 2026-10-28. To be run on the laptop.

## Question

Part 3a of `docs/write_up_plan.md` asks how the starting $k$ for annealing must grow with the vocabulary size $V$. Experiment 5 measured the $k_0$ that leaves a given share of features alive at initialisation. This experiment tests whether matching that share at the start also matches recovery at the end, given the same training per token. On the subset (T15: $k_0 = 128$, 1,024 epochs' worth) and at full scale (T11: $k_0 = 128$, about 1,300 epochs' worth), recovery was 75% and 70%.

## Design

- **Vocabularies:** the 5,000, 10,000 and 20,000 most frequent tokens, nested like the 2,000-token subset. $m = V$.
- **Starting $k$:** the $k_0$ that leaves 93% of features alive at initialisation, from Experiment 5: 42, 55 and 71.
- **Schedule:** the shape of T10. $k_0$ and the powers of 2 below it, down to 4, share the first 40% of training equally. Then $k = 2$ for 40% and $k = 1$ for the last 20%. For example, at $V = 5{,}000$: $k$ = 42, 32, 16, 8 and 4, then 2, then 1.
- **Training:** batch size 512 and the Gao learning rate, seed 0. The number of steps is $2V$: 10,000, 20,000 and 40,000. This holds the training per token fixed at 1,024 epochs' worth, as for 4,000 steps on the 2,000-token subset. A TopK feature gets gradient only when it is selected for a token in the minibatch, so the updates each feature receives scale with steps × batch / $V$, not with steps alone.
- **Measurement:** 40 evaluations per run, with decoder recovery, alive features and `n_top1` (the features that would survive a drop to $k = 1$ at that point; see Experiment 7).

## Expected cost

Scaled from the laptop's measured step times: about 14 minutes at 5,000 tokens, 53 minutes at 10,000 and 3.8 hours at 20,000. About 5 hours in all.

## How to run

```
caffeinate -i python experiments/exp8_topk_sizes/run.py
```

`--sizes` runs only some sizes, for example `--sizes 5000`. Results go to `runs/exp8/V<size>/`: a folder for the run (`eval_log.csv`, `result.json` and the final `model.pt`) and `summary.csv`, with one row per evaluation. Ctrl-C saves and stops, and the same command resumes, losing at most 10 minutes. A heartbeat every minute shows the ETA for all the chosen sizes. Checkpoints are deleted when a run finishes.

## Code

`experiments/exp8_topk_sizes/run.py` defines the schedules and calls `run_grids` in `sae_raw/grid.py`, which runs one grid per size.
