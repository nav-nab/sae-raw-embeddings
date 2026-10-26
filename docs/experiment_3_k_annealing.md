# Experiment 3: TopK with k-annealing

Status: done. Run on the laptop on 2026-10-19 to 20 with schedule O: 15,980 steps at batch size 4,096 in 10.2 hours. Results at the end.

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
| O | 128, 64, 32, 16, 8, 4 at 6.7% each; 2 for 40%; 1 for 20% | 1,891 | 1,523 | 1,499 | 1,250 |
| P | As B, starting at $k_0 = 128$ | 1,912 | 1,474 | 1,438 | 1,171 |

The starting $k$ matters most. Recovery rises steadily with $k_0$: 851, 1,171, 1,362 and 1,454 for $k_0$ = 8, 16, 32 and 64 under schedule B. A larger $k_0$ keeps more features alive before the step to $k = 1$.

The shape of the low end matters much less. A longer stage at $k = 2$, a stage at $k = 3$, or less time at $k = 1$ changes recovery by at most 40 tokens. With one seed, differences that small may be noise. In every run, the step from $k = 2$ to $k = 1$ still loses 15% to 20% of the alive features.

The best schedule, N, combines $k_0 = 64$ with a long stage at $k = 2$. It recovers 1,501 tokens, against 1,362 for B and 1,710 for the L1 reference with $\lambda = 0.3$.

Starting at $k_0 = 128$ (O and P) gives no further gain on the subset. At $k = 64$, 98% of the subset's features are already alive at initialisation, so a larger $k_0$ has little left to rescue.

## Choosing $k_0$ at full scale

The full table has a far larger share of features dead at initialisation than the subset at the same $k$.

| $k$ | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| Alive at initialisation, subset (of 2,000) | 93% | 98% | 99.8% | |
| Alive at initialisation, full table (of 50,257) | 76.7% | 86.5% | 93.3% | 97.3% |
| Step time at batch size 4096, full table (sandbox) | 4.7 s | 6.8 s | 9.6 s | |

On the subset, going from 93% to 98% alive at the start ($k_0$ = 32 to 64) raised recovery by about 90 tokens. At full scale, $k_0 = 128$ halves the features dead at the start, from 6,798 to 3,349, and brings the full table to where the subset starts at $k_0 = 32$. It costs about 10% of the steps in a 10-hour run. We therefore use schedule O, which matched N on the subset.

## Design

The same as Experiment 1, except for the encoder and the schedule:

- **Data:** the GPT-2 Small input embedding table, with $V = 50{,}257$ rows and $d = 768$. Tokens are sampled uniformly.
- **Model:** a TopK SAE with $m = V$ features. The encoder selects the $k$ largest pre-activations and applies a ReLU to them, so a selected feature with a pre-activation at or below zero is inactive. Decoder rows have unit norm, and the decoder bias is zero.
- **Loss:** plain MSE.
- **Initialisation:** as in Experiment 1, with the same seed. The initial parameters are saved to `init.pt`.
- **Training:** Adam with the Gao et al. learning rate, batch size 4096, one seed, a first budget of 10 hours on the laptop.
- **Schedule (O):** $k$ = 128, 64, 32, 16, 8 and 4, sharing the first 40% of training equally, then $k = 2$ for 40% and $k = 1$ for the last 20%.
- **Extension:** if the run needs longer, `--extend-steps` lengthens it. The part of the schedule not yet reached is stretched in proportion over the new length. An extension made during the $k = 2$ stage therefore lengthens that stage. An extension made after the run reaches $k = 1$ continues at $k = 1$, where dead features cannot revive.
- **Measurement:** as in Experiment 1. The current $k$, alive features and active features per token are logged at every evaluation. Loss, alive features and decoder recovery are measured at the end, once $k = 1$. Features dead at initialisation are judged at $k_0 = 128$.

## Decisions

- Fixed-$k$ TopK dropped in favour of $k$-annealing, based on the measurements above.
- First choice: schedule B (halving from $k_0 = 32$, reaching $k = 1$ at 80%), the best of the first three subset tests.
- Replaced on 2026-10-19 by schedule N, the best of the finer grid. It starts at $k_0 = 64$ and spends 40% of training at $k = 2$.
- Then replaced the same day by schedule O, which starts at $k_0 = 128$. It matched N on the subset and halves the features dead at initialisation at full scale.
- Budget: 10 hours first, then extend if needed.

## How to run

```
caffeinate -i python experiments/exp3_k_annealing/run.py
```

The script prepares the data if needed and times a few steps at each $k$ of the schedule, since steps at large $k$ are slower. It then chooses the number of steps that fits the 10-hour budget. Training runs with a checkpoint every 10 minutes and a heartbeat every minute that shows the current $k$. The script finishes with the analysis. Ctrl-C saves and stops, and the same command resumes. Results go to `runs/exp3/`. `--stages` changes the schedule, for example `--stages 64:0.08,32:0.08,16:0.08,8:0.08,4:0.08,2:0.4,1:0.2` for schedule N. Fractions are rescaled to sum to exactly 1. `--extend-steps N` lengthens the run, as described under Design. Run it once; to resume after an interrupt, use the plain command.

**Expected cost.** In the sandbox at full scale and batch size 4096, a step took 9.6 s at $k = 128$, 6.8 s at $k = 64$, 4.7 s at $k = 32$, about 3.3 s at $k = 16$ down to $k = 2$, and 2.7 s at $k = 1$. The mean over schedule O is about 3.9 s. Peak memory at $k = 32$ was about 3.7 GB. The laptop ran Top1 steps about 20% faster than the sandbox, so it should fit roughly 11,000 steps into 10 hours. The script measures this itself before training.

## Code

`experiments/exp3_k_annealing/run.py` describes the experiment and calls the shared runner in `sae_raw/run.py`. The model is `TopKSAE` in `sae_raw/model.py`; with $k = 1$ it matches `Top1SAE`. The schedule is `k_schedule` in `sae_raw/train.py`. `k_stages` lays it out over the run as a list of $k$ and last step, and `stretch_stages` lengthens it for an extension.

## Results

The run was not extended. The final model is a Top1 SAE, since the schedule ends at $k = 1$.

| | Experiment 1 (Top1 throughout) | Experiment 3 (schedule O) |
|---|---|---|
| Steps, batch size | 16,820 at 4,096 | 15,980 at 4,096 |
| Features alive at the end (of 50,257) | 259 | 36,354 (72%) |
| Relative loss | 0.654 | 0.073 |
| Tokens recovered at cosine > 0.9 | 16 | **35,261 (70%)** |
| Tokens recovered at cosine > 0.99 | | 28,502 (57%) |

Every token has an active feature at the end. 36,385 tokens have a mutual-nearest decoder row, with a median best cosine of 0.99997.

**Alive features along the run**, at the evaluations just before and after each drop in $k$:

| Point in the run | Step | $k$ | Alive features |
|---|---|---|---|
| Initialisation | 0 | 128 | 46,908 |
| Peak | 799 | 128 | 50,249 |
| After the drop to 32 | 2,397 | 32 | 48,409 |
| End of $k = 4$ | 6,392 | 4 | 47,147 |
| After the drop to 2 | 7,191 | 2 | 43,105 |
| End of $k = 2$ | 12,784 | 2 | 43,349 |
| After the drop to 1 | 13,583 | 1 | 36,351 |
| End | 15,980 | 1 | 36,354 |

### Findings

- **Annealing works at full scale.** Experiment 3 recovers 70% of the tokens, against 16 tokens for Top1 throughout. This matches the subset, where schedule O recovered 75%.
- **Surviving features are almost all correct.** 97% of the alive features are recovered tokens.
- **The shortfall comes from features dying at the drops in $k$.** The start at $k_0 = 128$ revived all 3,349 features dead at initialisation within 800 steps, and 50,249 features were alive at the peak. The drop to 32 cost about 1,800 features, and the drop to 2 about 4,000. The drop to $k = 1$ cost 6,998, or 16% of those alive, as on the subset (15% to 20%).
- **Dead features almost never revive.** Over the 6,400 steps at $k = 2$, the alive count rose by 244. After the drop to $k = 1$ it rose by 3.

### After the run: AuxK pilots (2026-10-20)

We tried to rescue the features lost at the drops with the auxiliary loss of Gao et al. (2024), AuxK, which trains dead features to reconstruct the residual. An argument predicts it fails at $k = 1$. A dead feature aligned with the residual of a token coded at cosine $\cos\theta$ scores about $\lvert x\rvert\sin\theta$ on that token, against $\lvert x\rvert\cos\theta$ for the incumbent. At $\cos\theta = 0.9$, that is $0.44\lvert x\rvert$ against $0.9\lvert x\rvert$, so the dead feature does not win the token.

On the subset with schedule O, two runs with standard AuxK recovered 1,476 and 1,478 tokens, against 1,499 without it. A variant that targets the whole token at $k = 1$ gave about 40 more tokens, which is close to the noise. AuxK was dropped. Even before the final drop it could not help much, because features die *at* the drops, not between them.
