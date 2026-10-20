# Write-up plan

Status: three-part structure agreed on 2026-10-20. More exploratory work comes before the write-up: ReLU tests on the subvocabulary for Part 1, and the measurements for Part 3.

## Thesis

With a dictionary of $m = V$ features trained on the raw token embedding table, theory says the SAE's features should be the token embeddings. For a Top1 SAE with unit-norm decoder rows and no decoder bias, the normalised embeddings are the zero-loss dictionary. For other sparse SAEs they remain the natural target.

The paper shows that whether training actually finds this dictionary depends heavily on the architecture and the training hyperparameters. Recovery is possible, but it is the exception. The setups that succeeded were found by trial and error.

Recovery throughout counts tokens with a mutual-nearest decoder row at cosine above 0.9 (and 0.99).

## Part 1: failure modes at $V = 2{,}000$

The tests on the 2,000 most frequent tokens ($m = 2{,}000$, seed 0). They show that recovery is desirable and expected, but whether it happens is very sensitive to the architecture and the training hyperparameters. Each test changes one choice and reports alive features, active features per token and recovery, out of 2,000.

The subset run with L1 at $\lambda = 0.3$ is the reference. It shows that recovery is possible on the subset, so a failure there cannot be blamed on the subset itself.

Budgets match within each section but not across them. The TopK tests ran at batch size 512 and the ReLU tests at 1,024.

### 1a. TopK SAEs

4,000 steps at batch size 512, except the fixed-$k$ runs, which used 3,000 steps.

| Test | Schedule | Alive at end | Active per token | Recovered |
|---|---|---|---|---|
| Top1 | $k = 1$ throughout | 382 | 1 | 227 |
| Fixed $k$ (3,000 steps) | $k$ = 2, 4, 8, 16, 32 | 558 to 1,778 | $k$ | 162, 73, 7, 0, 0 |
| Annealing, low start | $2 \to 1$; $4 \to 2 \to 1$ | 545; 750 | 1 | 295; 506 |
| Annealing, varying $k_0$ | Halving to 1 at 80%, $k_0$ = 8, 16, 32, 64, 128 | 1,025 to 1,482 | 1 | 851, 1,171, 1,362, 1,454, 1,438 |
| Annealing, varying the low end | From 32; longer $k = 2$, a $k = 3$ stage, or less time at $k = 1$ | 1,408 to 1,441 | 1 | 1,363 to 1,402 |
| Annealing, linear | Linear from 32 to 1 at 50% | 1,188 | 1 | 1,057 |
| **Annealing N** | 64, 32, 16, 8, 4 at 8% each; 2 for 40%; 1 for 20% | 1,532 | 1 | **1,501** |
| Annealing O | As N, starting at 128 | 1,523 | 1 | 1,499 |

- **Top1 fails** because most features are dead at initialisation and never receive gradient.
- **Fixed $k > 1$ fails** in the opposite way. Features stay alive, but every token uses all $k$ of them, so no feature lines up with a token.
- **$k$-annealing is a partial success.** A large starting $k$ keeps features alive, and recovery rises with $k_0$ until nearly all features are alive at the start. In every schedule, 15% to 20% of the alive features die at the final step from $k = 2$ to $k = 1$. The shape of the low end changes little. The best schedules recover 75% of the subset, against 86% for the reference.

### 1b. ReLU SAEs with an L1 penalty

4,000 steps at batch size 1,024.

| $\lambda$ | Alive at end | Active per token | Recovered |
|---|---|---|---|
| 0.03 | 2,000 | 79.5 | 3 |
| 0.1 | 2,000 | 5.0 | 1,055 |
| **0.3 (reference)** | 1,894 | 1.7 | **1,710** |
| 1 | 895 | 1.7 | 707 |

- **Too weak a penalty** leaves codes dense, so no feature lines up with a token.
- **Too strong a penalty** kills features early: half are dead within 500 steps at $\lambda = 1$.
- Only a narrow range of $\lambda$ works.
- **To do:** this section has only the $\lambda$ sweep so far. More tests are needed to show that L1 is as sensitive as TopK. Candidates are the learning rate, the batch size, and a longer run at $\lambda = 0.1$.

`docs/failure_modes.md` is the running source for this part.

## Part 2: success modes at GPT-2 scale

The full table, $V = 50{,}257$. The Top1 run of Experiment 1 is the baseline: 259 features alive at the end and 16 tokens recovered.

| | Experiment 3: $k$-annealing (schedule O) | Experiment 2: L1, $\lambda = 0.3$ |
|---|---|---|
| Training | 15,980 steps at batch 4,096, 10.2 h | 85,100 steps at batch 1,024, 46.6 h |
| Alive at end | 36,354 (72%) | 49,869 (99%) |
| Active features per token | 1 | 1.37 |
| Recovered at cosine > 0.9 | 35,261 (70.2%) | 46,590 (92.7%) |
| Recovered at cosine > 0.99 | 28,502 (56.7%) | 44,337 (88.2%) |
| Relative loss | 0.073 | 0.0066 |

### 2a. Experiment 3: TopK with $k$-annealing, a partial success

- **Schedule O.** $k$ = 128, 64, 32, 16, 8 and 4 share the first 40% of training, then $k = 2$ runs for 40% and $k = 1$ for 20%. The starting $k$ was chosen so that 93% of features are alive at initialisation (see Part 3).
- **Surviving features are almost all correct.** 35,261 of the 36,354 alive features are recovered tokens (97%), with a median best cosine of 0.99997.
- **The shortfall is features dying when $k$ drops.** Alive features peak at 50,249 at $k = 128$. About 4,000 die at the drop to $k = 2$, and about 7,000 more (16%) at the drop to $k = 1$.
- **The subset predicted the outcome.** Schedule O recovered 75% of the subset and 70% of the full table.

### 2b. Experiment 2: L1-penalised ReLU, the best result

- **Result.** 92.7% of tokens recovered at cosine above 0.9, with nearly every feature alive.
- **Training trajectory.** Codes start dense, with about 25,000 active features per token. They stall near 300 and then sparsify slowly towards one per token. Recovery only appears once codes are nearly one-hot. The run needed about four times its first 10-hour budget, so convergence time is part of the story.
- **Code structure.** 83% of tokens are coded by exactly one feature. The coefficient of that feature equals the embedding norm minus $\lambda/2$, the L1 shrinkage.
- **Where it falls short.** The unrecovered 7% lie in crowded regions of the embedding space. There a feature settles between two or three similar tokens, and other features make up the difference.

## Part 3: scaling laws

How the hyperparameters that make recovery work must change as the vocabulary grows.

### 3a. The starting $k$ for TopK

For $k$-annealing, what seems to matter is the share of features alive at initialisation. At the same $k$, that share falls as the vocabulary grows:

| $k$ | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| Alive at initialisation, $V = 2{,}000$ | 93% | 98% | 99.8% | |
| Alive at initialisation, $V = 50{,}257$ | 76.7% | 86.5% | 93.3% | 97.3% |

So keeping the same share alive at the start, and with it the same share of recovered tokens, requires $k_0$ to grow with $V$. Here, 93% alive takes $k_0 = 32$ at $V = 2{,}000$ and $k_0 = 128$ at $V = 50{,}257$. Both runs then recovered a similar share of tokens: 68% for B at $V = 2{,}000$ and 70% for O at full scale.

**Needed:** the alive share at initialisation at several vocabulary sizes in between, and annealing runs at a few of them. These would test whether the share at the start predicts recovery at the end, and give the relation between $k_0$ and $V$.

### 3b. The L1 coefficient $\lambda$ for ReLU

Open question: does $\lambda$ also need to change with $V$ to keep the same share of recovered tokens? So far, $\lambda = 0.3$ was best at $V = 2{,}000$ and also worked at full scale, but no other value was tried at full scale.

**Needed:** short $\lambda$ sweeps at several vocabulary sizes, scored by recovery. Full-scale runs are slow for L1 (Experiment 2 took 47 hours), so intermediate sizes will carry most of the weight.

## Analytic themes

Two themes run through all three parts and are best discussed together.

### The trade-off between sparsity and keeping features alive

Recovery needs both nearly one-hot codes and nearly all features alive. The two pull against each other:

- **Fixed $k$.** A small $k$ gives sparse codes but kills most features. A large $k$ keeps features alive but gives dense codes, and recovery is gone by $k = 16$.
- **L1 strength.** A weak $\lambda$ keeps every feature alive with dense codes. A strong $\lambda$ gives sparse codes but kills features early.
- **$k$-annealing.** Each drop in $k$ makes codes sparser and kills features: 15% to 20% at the final drop, on the subset and at full scale.
- **L1 at full scale handles the trade-off best.** In Experiment 2, active features per token fell from 24 to 1.37 between steps 18,500 and 85,100, and only 371 features died. In Experiment 3, the drops from $k = 4$ to $k = 1$ killed about 11,000.

A plausible reading is that L1 sparsifies gradually, one token at a time, while TopK sparsifies abruptly for all tokens at once.

### Very few features revive

Once a feature dies, it almost never comes back:

- **Experiment 1.** None of the 39,914 features dead at initialisation revived.
- **Experiment 3.** The 6,400 steps at $k = 2$ revived a net 244 of the roughly 7,000 features lost at the previous drops. After the drop to $k = 1$, the alive count rose by only 3 over the last 2,400 steps.
- **The exception is a large $k$ at the start.** Experiment 3 revived all 3,349 features dead at initialisation within 800 steps at $k = 128$. Revival happens when $k$ is raised well above what kept a feature dead, not with time.

This is why the starting conditions matter so much, and why the drops in $k$ are where TopK loses. The figures above are net changes in the alive count. Counting features that die and later revive would need per-feature activity at each evaluation, which the saved snapshots allow.
