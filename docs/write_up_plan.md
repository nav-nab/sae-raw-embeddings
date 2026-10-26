# Write-up plan

Status: restructured on 2026-10-26 into three parts: TopK SAEs, ReLU SAEs with an L1 penalty, and scaling laws. The experiments still needed are listed at the end.

## Thesis

With a dictionary of $m = V$ features trained on the raw token embedding table, the SAE's features should be the token embeddings. Whether training actually finds them depends mainly on the architecture.

- **TopK SAEs** lose features whenever $k$ is small or drops, and dead features almost never revive. The best TopK setup recovers 70% of GPT-2's tokens.
- **ReLU SAEs with an L1 penalty** recover the embeddings across a wide region of hyperparameters, provided they train long enough: 93% at full scale and 98% on a subvocabulary. Outside that region they fail in one of two ways: dense codes from too little training, or dead features from too much pressure.

## Setting and measurement

- **Data.** The GPT-2 Small input embedding table: $V = 50{,}257$ rows in $d = 768$ dimensions. Pilots use a subvocabulary of the 2,000 GPT-2 tokens that occur most often in English Wikipedia (about 200 million tokens counted, snapshot `20231101.en`). Frequency only picks the subset. Training always samples tokens uniformly.
- **Model.** $m = V$ features, decoder rows of unit norm, no decoder bias. Initialisation uses no information from the embeddings: random Gaussian decoder rows, encoder copied from the decoder, zero encoder bias.
- **Training.** Adam, with the Gao et al. learning rate rule $2\times10^{-4}/\sqrt{m/2^{14}}$ as the base value. One seed per run.
- **Recovery.** A token is recovered if some decoder row is its mutual nearest neighbour at cosine above 0.9 (and, more strictly, above 0.99).

## Part 1: TopK SAEs

### Theory: the zero-loss dictionary

A Top1 SAE ($k = 1$) reconstructs each token as one decoder row times a non-negative coefficient. With unit-norm rows and no decoder bias, a token $x$ is reconstructed exactly only if some row equals $x/\lvert x\rvert$. With $m = V$, the zero-loss dictionary is therefore the set of normalised embeddings, unique up to permutation as long as no two embeddings point the same way. For $k > 1$ this uniqueness is lost: a token can be rebuilt from several features, and many other dictionaries also reach zero loss.

### The trade-off between sparsity and keeping features alive

A TopK feature receives gradient only on tokens where it is among the top $k$. This creates a trade-off:

- **A small $k$** gives the sparse codes that make the embeddings the target, but most features are never selected. At $k = 1$, only 20.6% of features are alive at initialisation on the full table. A feature that is never selected gets no gradient, so it stays dead.
- **A large $k$** keeps features alive, but every token then uses all $k$ slots. No feature has to line up with a single token.
- **$k$-annealing** tries to get both, by starting with a large $k$ and lowering it to 1. Each drop in $k$ is a sudden increase in sparsity for every token at once, and each one kills features.

### 1a. Failure modes

Subset runs ($V = m = 2{,}000$), 4,000 steps at batch size 512 unless marked. Recovered counts are out of 2,000.

| Test | Alive at end | Recovered > 0.9 | Recovered > 0.99 |
|---|---|---|---|
| Top1 | 382 | 227 | 113 |
| Fixed $k$ = 2, 4, 8, 16, 32 (3,000 steps) | 558 to 1,778 | 162, 73, 7, 0, 0 | 13, 0, 0, 0, 0 |
| Annealing with a low start: $2 \to 1$; $4 \to 2 \to 1$ | 545; 750 | 295; 506 | 136; 250 |
| Annealing from $k_0$ = 8, 16 (halving, $k = 1$ at 80%) | 1,025; 1,265 | 851; 1,171 | 530; 849 |

- **Top1 fails** because most features are dead at initialisation and never revive. At full scale (Experiment 1), none of the 39,914 features dead at initialisation revived, 259 features were alive at the end, and 16 tokens were recovered.
- **Fixed $k > 1$ fails** in the opposite way. Features stay alive, but codes use all $k$ slots, and recovery is gone by $k = 16$.
- **Annealing from a small $k_0$ fails** because too few features are alive at the start.
- **An auxiliary loss does not rescue the dead features.** AuxK (Gao et al., 2024) trains dead features to reconstruct the residual. An argument predicts this should fail at $k = 1$. A dead feature aligned with the residual of a token coded at cosine $\cos\theta$ scores about $\lvert x\rvert\sin\theta$ on that token, against $\lvert x\rvert\cos\theta$ for the incumbent: at $\cos\theta = 0.9$, $0.44\lvert x\rvert$ against $0.9\lvert x\rvert$. On the subset with schedule O, standard AuxK gave nothing (1,476 and 1,478 recovered, against 1,499 without it). A variant that targets the whole token at $k = 1$ gave about 40 more tokens, which is close to the noise. Even before the drops it could not help, because features die *at* the drops.

### 1b. Successes: annealing from a large $k$

| Run | Schedule | Alive at end | Recovered > 0.9 | Recovered > 0.99 |
|---|---|---|---|---|
| Subset, B | Halving from 32, $k = 1$ at 80% | 1,414 | 1,362 (68%) | 1,068 |
| Subset, N | 64, 32, 16, 8, 4 at 8% each; 2 for 40%; 1 for 20% | 1,532 | 1,501 (75%) | 1,267 |
| Subset, O | As N, starting at 128 | 1,523 | 1,499 (75%) | 1,250 |
| **Full scale, Experiment 3** | O, 15,980 steps at batch 4,096 | 36,354 (72%) | **35,261 (70%)** | 28,502 (57%) |

- **Surviving features are almost all correct.** In Experiment 3, 97% of the alive features are recovered tokens, with a median best cosine of 0.99997.
- **The shortfall is entirely features dying at the drops.** Alive features peak at 50,249 at $k = 128$. About 4,000 die at the drop to $k = 2$, and about 7,000 more (16%) at the drop to $k = 1$. On the subset, too, every schedule loses 15% to 20% at the final drop.
- **Dead features almost never revive.** Over the 6,400 steps at $k = 2$, the alive count rose by only 244. After the drop to $k = 1$, it rose by 3. The one exception is a large $k$ at the start: $k_0 = 128$ revived all 3,349 features dead at initialisation within 800 steps.
- **What matters most is the starting $k$.** On the subset, recovery rises with $k_0$ until nearly every feature is alive at the start. The shape of the low end changes it by at most 40 tokens.
- **The subset predicts full scale well.** Schedule O recovered 75% of the subset and 70% of the full table.

## Part 2: ReLU SAEs with an L1 penalty

### Theory: no zero-loss dictionary, but the embeddings are still the natural target

The loss is $\lvert x - \hat{x}\rvert^2 + \lambda\sum_j f_j$, with codes $f_j \ge 0$ and $\hat{x} = \sum_j f_j d_j$. With $\lambda > 0$, no dictionary reaches zero loss. The embeddings remain the natural target, for two reasons:

- **One-hot codes are the cheapest.** By the triangle inequality, $\sum_j f_j \ge \lvert \hat{x}\rvert$ for unit-norm rows, with equality only if every active row points along $\hat{x}$. So any reconstruction costs the least L1 penalty when it uses a single feature aligned with it.
- **The embeddings minimise each token's loss.** For one token, the loss over all reconstructions $y$ is at least $\min_y \lvert x - y\rvert^2 + \lambda\lvert y\rvert$. The minimum is at $y$ along $x$ with $\lvert y\rvert = \lvert x\rvert - \lambda/2$, giving $\lambda\lvert x\rvert - \lambda^2/4$. A feature equal to $x/\lvert x\rvert$, with coefficient $\lvert x\rvert - \lambda/2$, attains it. With $m = V$, every token can have its own such feature. Then every token is at its own minimum, so the normalised embeddings minimise the total loss. Experiment 2 shows exactly this shrinkage: the coefficient of a token's feature is its norm minus 0.15, which is $\lambda/2$.

The second argument assumes the encoder can produce exact one-hot codes: each feature must be inactive on every other token. Whether a ReLU encoder can do this for the GPT-2 embeddings is still to be checked (see the next experiments).

### Empirical findings: where recovery happens

On the subset, Experiment 4 maps recovery over $\lambda$ (0.03 to 1), the learning rate (¼ to 4 times the Gao rule) and the batch size (256 to 4,096), at up to 8 million training tokens, with one seed.

| $\lambda$ | Best recovery once converged (of 2,000) | Training needed to converge (steps at the Gao rate) |
|---|---|---|
| 0.03 | 1,959 (98%), possibly still rising | 32,000 or more |
| 0.1 | about 1,890 | about 8,000 |
| 0.3 | about 1,750 | about 4,000 |
| 1 | about 1,350 | about 2,000 |

There are three regimes:

- **Undertrained: dense codes.** At a small $\lambda$, a low learning rate or too few steps, every feature stays alive, but tokens still use many features each. This is a slow success, not a true failure. These runs are on the way to recovery and get there with more steps. An earlier sweep at a fixed 4,000 steps had wrongly classed $\lambda = 0.03$ as a failure for this reason.
- **Successful.** Across $\lambda$ from 0.03 to 0.5, at learning rates up to the Gao rule, enough training gives over 80% recovery. The best results come from the smallest $\lambda$, at the cost of the longest training.
- **Over-pressured: dead features.** A large $\lambda$ or a high learning rate kills features early. At 4 times the Gao rate with $\lambda = 1$, only 3 to 7 features survive. These deaths happen in the first half-million tokens, and more training does not undo them.

Two rules describe the successful region:

- **Training runs on learning rate × steps, not tokens seen.** Runs with the same product reach similar sparsity, whatever their batch size. Adam moves each weight by about the learning rate per step, so a larger batch gives a cleaner gradient but no faster progress.
- **A weaker penalty recovers more, but needs longer.** A larger $\lambda$ kills more features on the way to one-hot codes, so its final recovery is lower. A smaller $\lambda$ kills fewer, but sparsifies more slowly. This is the ReLU form of the trade-off in Part 1.

At full scale, Experiment 2 ($\lambda = 0.3$, batch 1,024) recovered 46,590 tokens at cosine above 0.9 (93%) and 44,337 above 0.99 (88%). 49,869 features were alive, and there were 1.37 active features per token. It needed 85,100 steps and 47 hours. Codes went from about 25,000 active features per token to 1.37 while only about 400 features died. This gradual sparsification, one token at a time, is the contrast with TopK's abrupt drops. The unrecovered 7% lie in crowded regions of the embedding space, where one feature settles between two or three similar tokens.

## Part 3: scaling laws

How the settings that make recovery work must change as the vocabulary grows.

### 3a. The starting $k$ for TopK

At the same $k$, fewer features are alive at initialisation in a larger vocabulary:

| $k$ | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| Alive at initialisation, $V = 2{,}000$ | 93% | 98% | 99.8% | |
| Alive at initialisation, $V = 50{,}257$ | 76.7% | 86.5% | 93.3% | 97.3% |

Keeping the same share alive at the start requires $k_0$ to grow with $V$. Here, 93% alive takes $k_0 = 32$ at $V = 2{,}000$ and $k_0 = 128$ at $V = 50{,}257$. Matched this way, the two runs recovered similar shares: 68% (schedule B on the subset) and 70% (Experiment 3). This rests on two vocabulary sizes so far.

### 3b. The L1 coefficient and training length for ReLU

Open questions: how does the training needed for a given $\lambda$ grow with $V$, and does the best $\lambda$ change? On the subset, $\lambda = 0.03$ beat 0.3 given enough training. At full scale only $\lambda = 0.3$ has been run, and it already took 47 hours.

## Next experiments

In order of priority:

1. **Alive share at initialisation against $k$ at intermediate vocabulary sizes** (Part 3a). For example $V$ = 5,000, 10,000 and 20,000, plus the two sizes we have. No training is needed, so this takes minutes in the sandbox. It gives the relation between $k_0$ and $V$ for a fixed alive share.
2. **Annealing runs at two or three intermediate sizes** (Part 3a), with $k_0$ chosen to give the same alive share at the start. This tests whether the alive share at the start predicts recovery at the end. Each run is short compared with Experiment 3.
3. **ReLU runs at the same intermediate sizes** (Part 3b). These would use a few values of $\lambda$, trained long enough to converge, to measure how training length and the best $\lambda$ change with $V$. The subset grid showed that progress depends on learning rate × steps, so the runs can use a small batch and the Gao rate.
4. **A check of the ReLU theory** (Part 2). Test whether a ReLU encoder can produce exact one-hot codes for the GPT-2 embeddings, by constructing the encoder from the embeddings and choosing biases. This is a short calculation, not a training run.
5. **Write-up housekeeping.** Add the Experiment 3 results to its doc and to `docs/failure_modes.md`.

One optional item is a TopK check of learning rate and batch size, mirroring the ReLU grid. It would show whether TopK's failures are also partly undertraining. I'd leave it out unless a reader is likely to ask, because the TopK losses happen at the drops in $k$, which more training at $k = 1$ cannot reverse.
