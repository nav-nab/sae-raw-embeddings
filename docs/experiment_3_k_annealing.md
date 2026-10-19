# Experiment 3: TopK with k-annealing

Status: ready to run. Schedule chosen (schedule B below) and script written.

## Motivation

Experiment 1 trained a Top1 SAE ($k = 1$) and failed. No dead feature revived, and the alive features collapsed from 10,343 to 259 (`docs/experiment_1_results.md`). Our reading was that the $k = 1$ cutoff is too harsh: only the winning feature for each token receives gradient.

A fixed $k$ pulls in two directions. A large $k$ gives gradient to many features, so fewer die. A small $k$, ideally $k = 1$, is what makes the raw embeddings the only zero-loss dictionary. With $k > 1$, a token can be rebuilt from several features, so many other zero-loss dictionaries exist.

We first planned an experiment with a fixed $k > 1$ and measured the trade-off before running it. The measurements below showed that no fixed $k$ does both jobs, so that experiment was dropped.

$k$-annealing tries to get both. Training starts with a large $k$, so that features stay alive and spread out over the embeddings. $k$ is then lowered during training until it reaches 1, so that each token is pushed to use a single feature. If features have found their tokens by then, the final Top1 model could be close to the embedding dictionary.

## Measurements with fixed k

**Alive features at initialisation (full table, seed 0).** A feature is alive if it is among the top $k$ with a positive pre-activation for at least one token.

| $k$ | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Alive at initialisation | 20.6% | 29.1% | 39.7% | 51.9% | 64.6% | 76.7% | 86.5% |

**Survival and recovery after training.** These runs used the 2,000 most frequent tokens with $m = 2{,}000$, 3,000 steps at batch size 512. A token is recovered if it has a mutual-nearest decoder row at cosine above 0.9.

| $k$ | Alive at end (of 2,000) | Relative loss | Active features per token | Recovered |
|---|---|---|---|---|
| 1 | 382 | 0.448 | 1 | 222 |
| 2 | 558 | 0.317 | 2 | 162 |
| 4 | 792 | 0.197 | 4 | 73 |
| 8 | 1,111 | 0.098 | 8 | 7 |
| 16 | 1,478 | 0.033 | 16 | 0 |
| 32 | 1,778 | 0.008 | 32 | 0 |

Raising $k$ keeps more features alive and lowers the loss, but recovery falls and is gone by $k = 16$. Every run used all $k$ slots for every token. Once a token can be rebuilt from several features, the loss rewards doing so, and nothing pulls the code back to one feature per token. With $k \ge 2$, nearly all features dead at initialisation revived at some point, but many died again before the end.

These are subset results. The full table has 50,257 features in the same 768 dimensions, so the share of features that stays alive may differ at full scale.

## Schedule tests on the subset

These runs used the same 2,000-token subset with $m = 2{,}000$, 4,000 steps at batch size 512 and the same seed. Every annealed run started at $k_0 = 32$ and ended at $k = 1$, so all are judged as Top1 models at the end.

| Run | Schedule | Alive at end (of 2,000) | Recovered at cosine > 0.9 | Recovered at cosine > 0.99 | Relative loss |
|---|---|---|---|---|---|
| Baseline | Fixed $k = 1$ | 382 | 227 | 113 | 0.447 |
| A | Halving $32 \to 16 \to 8 \to 4 \to 2 \to 1$, reaching 1 at 50% of training | 1,377 | 1,325 | 1,011 | 0.085 |
| B | Same halving, reaching 1 at 80% of training | 1,414 | 1,362 | 1,068 | 0.079 |
| C | Linear from 32 to 1, reaching 1 at 50% of training | 1,188 | 1,057 | 776 | 0.132 |

Annealing works on the subset. The best schedule, B, recovers 1,362 tokens at cosine above 0.9, six times the baseline.

The trajectories show where features are lost. About 1,775 features stay alive through the high-$k$ phase in every run. Most of the loss happens at the final step from $k = 2$ to $k = 1$, where B drops from 1,731 to 1,412 alive features. The linear schedule passes through the small values of $k$ fastest, and it does worst.

## Finer schedule grid on the subset (2026-10-19)

Same budget as the tests above: 4,000 steps at batch size 512, seed 0. Each schedule is given as the share of training spent at each $k$. "Alive before $k = 1$" is the alive count at the last evaluation before $k$ reaches 1.

| Run | Schedule | Alive before $k = 1$ | Alive at end | Recovered at cosine > 0.9 | Recovered at cosine > 0.99 |
|---|---|---|---|---|---|
| B | Halving from 32, reaching 1 at 80% | 1,731 | 1,414 | 1,362 | 1,068 |
| D | As B, reaching 1 at 95% | 1,731 | 1,408 | 1,363 | 1,065 |
| G | 32, 16, 8, 4 at 10% each; 2 for 40%; 1 for 20% | 1,714 | 1,441 | 1,402 | 1,115 |
| H | 32, 16, 8, 4 at 5% each; 2 for 60%; 1 for 20% | 1,654 | 1,440 | 1,393 | 1,119 |
| I | As B, with a stage at $k = 3$ between 4 and 2 | 1,758 | 1,420 | 1,368 | 1,089 |
| J | 32, 16, 8, 4 at 5% each; 2 for 75%; 1 for 5% | 1,654 | 1,433 | 1,387 | 1,101 |
| E | As B, starting at $k_0 = 8$ | 1,101 | 1,025 | 851 | 530 |
| F | As B, starting at $k_0 = 16$ | 1,457 | 1,265 | 1,171 | 849 |
| K | 4 for 20%, 2 for 60%, 1 for 20% | 781 | 750 | 506 | 250 |
| L | 2 for 80%, 1 for 20% | 557 | 545 | 295 | 136 |
| M | As B, starting at $k_0 = 64$ | 1,884 | 1,482 | 1,454 | 1,184 |
| N | 64, 32, 16, 8, 4 at 8% each; 2 for 40%; 1 for 20% | 1,860 | 1,532 | **1,501** | **1,267** |

The starting $k$ matters most. Recovery rises steadily with $k_0$: 851, 1,171, 1,362 and 1,454 for $k_0$ = 8, 16, 32 and 64 under schedule B. A larger $k_0$ keeps more features alive before the step to $k = 1$.

The shape of the low end matters much less. A longer stage at $k = 2$, a stage at $k = 3$, or less time at $k = 1$ changes recovery by at most 40 tokens. With one seed, differences that small may be noise. In every run, the step from $k = 2$ to $k = 1$ still loses 15% to 20% of the alive features.

The best schedule, N, combines $k_0 = 64$ with a long stage at $k = 2$. It recovers 1,501 tokens, against 1,362 for B and 1,710 for the L1 reference with $\lambda = 0.3$. The trend in $k_0$ has not levelled off at 64.

## Design

The same as Experiment 1, except for the encoder and the schedule:

- **Data:** the GPT-2 Small input embedding table, with $V = 50{,}257$ rows and $d = 768$. Tokens are sampled uniformly.
- **Model:** a TopK SAE with $m = V$ features. The encoder selects the $k$ largest pre-activations and applies a ReLU to them, so a selected feature with a pre-activation at or below zero is inactive. Decoder rows have unit norm, and the decoder bias is zero.
- **Loss:** plain MSE.
- **Initialisation:** as in Experiment 1, with the same seed. The initial parameters are saved to `init.pt`.
- **Training:** Adam with the Gao et al. learning rate, batch size 4096, one seed, a budget of about 10 hours on the laptop.
- **Schedule (B):** $k$ halves from $k_0 = 32$ through 16, 8, 4 and 2, in five equal stages, and reaches 1 at 80% of training. It stays at 1 for the last 20%.
- **Measurement:** as in Experiment 1. The current $k$, alive features and active features per token are logged at every evaluation. Loss, alive features and decoder recovery are measured at the end, once $k = 1$. Features dead at initialisation are judged at $k_0 = 32$.

## Decisions

- Fixed-$k$ TopK dropped in favour of $k$-annealing, based on the measurements above.
- $k_0 = 32$.
- Schedule B: halving, reaching $k = 1$ at 80% of training. It did best of the three schedules tested on the subset.
- To revisit later: spending longer at $k = 2$. Most features that die are lost at the final step from $k = 2$ to $k = 1$, so a longer stage at $k = 2$ might keep more of them.

## How to run

```
caffeinate -i python experiments/exp3_k_annealing/run.py
```

The script prepares the data if needed and times a few steps at each $k$ of the schedule, since steps at large $k$ are slower. It then chooses the number of steps that fits the 10-hour budget. Training runs with a checkpoint every 10 minutes and a heartbeat every minute that shows the current $k$. The script finishes with the analysis. Ctrl-C saves and stops, and the same command resumes. Results go to `runs/exp3/`. `--k0`, `--end-frac` and `--shape` change the schedule.

**Expected cost.** In the sandbox at full scale and batch size 4096, a step took 4.6 s at $k = 32$, about 3.3 s at $k = 16$ down to $k = 2$, and 2.7 s at $k = 1$. The mean over schedule B is 3.4 s, about 25% more than a Top1 step. Peak memory was about 3.7 GB. The laptop ran Top1 steps about 20% faster than the sandbox, so it should fit roughly 13,000 steps into 10 hours. The script measures this itself before training.

## Code

`experiments/exp3_k_annealing/run.py` describes the experiment and calls the shared runner in `sae_raw/run.py`. The model is `TopKSAE` in `sae_raw/model.py`; with $k = 1$ it matches `Top1SAE`. The schedule is `k_schedule` in `sae_raw/train.py`.
