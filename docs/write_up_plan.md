# Write-up plan

Status: framing agreed; more exploratory work on the subset comes before the write-up.

## Thesis

With a dictionary of $m = V$ features trained on the raw token embedding table, theory says the SAE's features should be the token embeddings. For a Top1 SAE with unit-norm decoder rows and no decoder bias, the normalised embeddings are the zero-loss dictionary. For other sparse SAEs they remain the natural target.

The paper shows that whether training actually finds this dictionary depends heavily on the architecture and the training hyperparameters. Recovery is possible, but it is the exception. The setup that succeeded was found by trial and error.

## Part 1: the success mode

The full GPT-2 Small run of Experiment 2: an L1-penalised ReLU SAE with $\lambda = 0.3$, batch size 1024 and the Gao et al. learning rate, trained for 85,100 steps from random initialisation.

- **Result.** 46,590 of 50,257 tokens recovered at cosine above 0.9 (92.7%), and 44,337 above 0.99 (88.2%). The median best cosine is 0.99997. 49,869 features are alive, and there are 1.37 active features per token.
- **Training trajectory.** Codes start dense, with about 25,000 active features per token. They stall near 300 and then sparsify slowly towards one per token. Recovery only appears once codes are nearly one-hot. The run needed about four times its first 10-hour budget, so convergence time is part of the story.
- **Why it is a success.** 83% of tokens are coded by exactly one feature. The coefficient of that feature equals the embedding norm minus $\lambda/2$, the L1 shrinkage.
- **Where it falls short.** The unrecovered 7% lie in crowded regions of the embedding space. There a feature settles between two or three similar tokens, and other features make up the difference.

## Part 2: the failure modes

The tests on the 2,000-token subvocabulary ($m = 2{,}000$, seed 0), showing that convergence to the embeddings is rare. Each test changes one choice and reports alive features, active features per token and recovery. Recovery counts tokens with a mutual-nearest decoder row at cosine above 0.9, out of 2,000.

The subset run of the success mode (L1, $\lambda = 0.3$) is the reference. It shows that recovery is possible on the subset, so a failure there cannot be blamed on the subset itself.

Part 2 has two sections, one per architecture. Budgets match within each section but not across them. The TopK tests ran at batch size 512 and the ReLU tests at 1,024.

### 2a. TopK SAEs

Subset runs, 4,000 steps at batch size 512 unless marked. The fixed-$k$ runs used 3,000 steps.

| Test | Schedule | Alive at end | Active per token | Recovered |
|---|---|---|---|---|
| Top1 | $k = 1$ throughout | 382 | 1 | 227 |
| Fixed $k$ (3,000 steps) | $k$ = 2, 4, 8, 16, 32 | 558 to 1,778 | $k$ | 162, 73, 7, 0, 0 |
| Annealing, low start | $2 \to 1$; $4 \to 2 \to 1$ | 545; 750 | 1 | 295; 506 |
| Annealing, varying $k_0$ | Halving to 1 at 80%, $k_0$ = 8, 16, 32, 64 | 1,025 to 1,482 | 1 | 851, 1,171, 1,362, 1,454 |
| Annealing, varying the low end | From 32; longer $k = 2$, a $k = 3$ stage, or less time at $k = 1$ | 1,408 to 1,441 | 1 | 1,363 to 1,402 |
| Annealing, linear | Linear from 32 to 1 at 50% | 1,188 | 1 | 1,057 |
| **Annealing N** | 64, 32, 16, 8, 4 at 8% each; 2 for 40%; 1 for 20% | 1,532 | 1 | **1,501** |
| Annealing O | As N, starting at 128 | 1,523 | 1 | 1,499 |

- **Top1 fails** because most features are dead at initialisation and never receive gradient. Experiment 1 is the full-scale version: none of the 39,914 dead features revived, and 16 tokens were recovered.
- **Fixed $k > 1$ fails** in the opposite way. Features stay alive, but every token uses all $k$ of them, so no feature lines up with a token.
- **$k$-annealing is a partial success.** A large starting $k$ keeps features alive, and recovery rises steadily with $k_0$. In every schedule, 15% to 20% of the alive features die at the final step from $k = 2$ to $k = 1$. The shape of the low end changes little. The best schedule, N, recovers 75% of the subset, against 86% for the reference.
- **Full scale:** Experiment 3 runs schedule O on the whole vocabulary. O matches N on the subset, and at full scale it halves the features dead at initialisation. Its result will close this section.

### 2b. ReLU SAEs with an L1 penalty

Subset runs, 4,000 steps at batch size 1,024.

| $\lambda$ | Alive at end | Active per token | Recovered |
|---|---|---|---|
| 0.03 | 2,000 | 79.5 | 3 |
| 0.1 | 2,000 | 5.0 | 1,055 |
| **0.3 (reference)** | 1,894 | 1.7 | **1,710** |
| 1 | 895 | 1.7 | 707 |

- **Too weak a penalty** leaves codes dense, so no feature lines up with a token.
- **Too strong a penalty** kills features early: half are dead within 500 steps at $\lambda = 1$.
- Only a narrow range of $\lambda$ works. Experiment 2 shows that the working value also needs a long run at full scale: it reached one-hot codes only after about four times its planned budget.
- **To do:** this section has only the $\lambda$ sweep so far. More tests are needed to show that convergence is rare for L1 too. Candidates are the learning rate, the batch size, and a longer run at $\lambda = 0.1$.

`docs/failure_modes.md` is the running source for this part.
