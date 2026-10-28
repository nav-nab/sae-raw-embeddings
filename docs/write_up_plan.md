# Write-up plan

Status: restructured on 2026-10-26 into three parts: TopK SAEs, ReLU SAEs with an L1 penalty, and scaling laws. The experiments still needed are listed at the end.

## Thesis

With a dictionary of $m = V$ features trained on the raw token embedding table, the SAE's features should be the token embeddings. Whether training actually finds them depends mainly on the architecture.

- **TopK SAEs** lose features whenever $k$ is small or drops, and dead features almost never revive. The best TopK setup recovers 70% of GPT-2's tokens.
- **ReLU SAEs with an L1 penalty** recover the embeddings across a wide region of hyperparameters, provided they train long enough: 93% at full scale and 98% on a subvocabulary. Outside that region they fail in one of two ways: dense codes from too little training, or dead features from too much pressure.
- **For ReLU SAEs, recovery does not depend on the vocabulary size.** Given the same training per token, recovery as a function of $\lambda$ is the same at 2,000, 5,000, 10,000 and 20,000 tokens, to within a few points, and consistent with the full table. The number of steps needed grows in proportion to $V$ (at a fixed batch size, the same epochs' worth). So the subvocabulary behaves as a faithful small-scale model of the full problem, and results found there carry over. This is a conceptual theme of the write-up (Part 3b).

## Setting and measurement

- **Data.** The GPT-2 Small input embedding table: $V = 50{,}257$ rows in $d = 768$ dimensions. Pilots use a subvocabulary of the 2,000 GPT-2 tokens that occur most often in English Wikipedia (about 200 million tokens counted, snapshot `20231101.en`). Frequency only picks the subset. Training always samples tokens uniformly.
- **Model.** $m = V$ features, decoder rows of unit norm, no decoder bias. Initialisation uses no information from the embeddings: random Gaussian decoder rows, encoder copied from the decoder, zero encoder bias.
- **Training.** Adam, with a constant learning rate whose base value is $2\times10^{-4}/\sqrt{m/2^{14}}$. One seed per run. These docs call this base value "the Gao rule" or "the Gao rate". It is about $5.7\times10^{-4}$ on the subvocabulary and about $1.1\times10^{-4}$ at full scale.

**What "the Gao rule" means.** Checked against the paper on 2026-10-26. Gao et al. (2024, "Scaling and evaluating sparse autoencoders", Section 3.1.2) report that the largest learning rate that converges for their TopK SAEs scales as $1/\sqrt{n}$, where $n$ is the number of latents. They choose learning rates by sweeping at small scales and extrapolating that trend. They do not state a constant: the values $2\times10^{-4}$ and $2^{14}$ do not appear in the paper or its appendices. The constant is our own choice of normalisation. It matches the rule built into the EleutherAI `sparsify` library, which follows Gao et al.'s recipe, though its exact formula was not checked here. The write-up should therefore say that the learning rate scales as $1/\sqrt{m}$ following Gao et al., and that we normalise it to $2\times10^{-4}$ at $m = 2^{14}$. It should not say that Gao et al. give this value. Two further caveats: the scaling was derived for TopK SAEs on language-model activations, and we also use it for ReLU SAEs on the embedding table.
- **Recovery.** A token is recovered if some decoder row is its mutual nearest neighbour at cosine above 0.9 (and, more strictly, above 0.99).
- **Outcome colours in the run tables.** 🟢 success: at least 90% of tokens recovered at cosine above 0.9. 🟡 partial success: 50% to 90%. 🔴 failure: under 50%. Runs carry tags (T1, T2, … for TopK; R1, R2, … for ReLU) so the text can refer to them.

## Part 1: TopK SAEs

We trained TopK SAEs in three regimes:

1. **Fixed $k$**, either $k = 1$ (Top1) or a larger fixed $k$.
2. **$k$-annealing**: $k$ starts large and is lowered in stages to 1 during training.
3. **$k$-annealing with AuxK**: as regime 2, with the auxiliary loss of Gao et al. (2024) to revive dead features.

### Theory: the zero-loss dictionary

A Top1 SAE ($k = 1$) reconstructs each token as one decoder row times a non-negative coefficient. With unit-norm rows and no decoder bias, a token $x$ is reconstructed exactly only if some row equals $x/\lvert x\rvert$. With $m = V$, the zero-loss dictionary is therefore the set of normalised embeddings, unique up to permutation as long as no two embeddings point the same way. For $k > 1$ this uniqueness is lost: a token can be rebuilt from several features, and many other dictionaries also reach zero loss.

### The trade-off between sparsity and keeping features alive

A TopK feature receives gradient only on tokens where it is among the top $k$. This creates a trade-off:

- **A small $k$** gives the sparse codes that make the embeddings the target, but most features are never selected. At $k = 1$, only 20.6% of features are alive at initialisation on the full table. A feature that is never selected gets no gradient, so it stays dead.
- **A large $k$** keeps features alive, but every token then uses all $k$ slots. No feature has to line up with a single token.
- **$k$-annealing** tries to get both, by starting with a large $k$ and lowering it to 1. Each drop in $k$ is a sudden increase in sparsity for every token at once, and each one kills features.

### Results

Subset runs have $V = m = 2{,}000$ and 4,000 steps at batch size 512, unless marked. Every annealed run ends at $k = 1$. The "long $k = 2$ stage" schedules halve $k$ from $k_0$ down to 4 in equal stages over the first 40% of training, then hold $k = 2$ for 40% and $k = 1$ for the last 20%. The "halving" schedules halve $k$ from $k_0$ in equal stages and reach $k = 1$ at 80% of training.

| # | Outcome | Run | Scale and training | Active features per token at end | Alive at end | Recovered > 0.9 | Recovered > 0.99 |
|---|---|---|---|---|---|---|---|
| | | **Fixed $k$** | | | | | |
| T1 | 🔴 | Top1 (Experiment 1) | Full table, 16,820 steps at batch 4,096 | 1 | 259 | 16 (0.03%) | 6 |
| T2 | 🔴 | Top1 | Subset | 1 | 382 | 227 (11%) | 113 |
| T3 | 🔴 | Fixed $k$ = 2, 4, 8, 16, 32 | Subset, 3,000 steps | $k$ | 558 to 1,778 | 162, 73, 7, 0, 0 | 13, 0, 0, 0, 0 |
| | | **$k$-annealing** | | | | | |
| T4 | 🔴 | 2 for 80%, then 1 | Subset | 1 | 545 | 295 (15%) | 136 |
| T5 | 🔴 | 4 for 20%, 2 for 60%, then 1 | Subset | 1 | 750 | 506 (25%) | 250 |
| T6 | 🔴 | Halving from $k_0 = 8$ | Subset | 1 | 1,025 | 851 (43%) | 530 |
| T7 | 🟡 | Halving from $k_0 = 16$ | Subset | 1 | 1,265 | 1,171 (59%) | 849 |
| T8 | 🟡 | Halving from $k_0 = 32$ | Subset | 1 | 1,414 | 1,362 (68%) | 1,068 |
| T9 | 🟡 | Long $k = 2$ stage from $k_0 = 64$ | Subset | 1 | 1,532 | 1,501 (75%) | 1,267 |
| T10 | 🟡 | Long $k = 2$ stage from $k_0 = 128$ | Subset | 1 | 1,523 | 1,499 (75%) | 1,250 |
| T11 | 🟡 | Long $k = 2$ stage from $k_0 = 128$ (Experiment 3) | Full table, 15,980 steps at batch 4,096 | 1 | 36,354 | **35,261 (70%)** | 28,502 (57%) |
| | | **$k$-annealing with AuxK** (as T10) | | | | | |
| T12 | 🟡 | Standard AuxK, weight $\alpha$ = 1/32, 1/8 | Subset | 1 | 1,507; 1,516 | 1,476; 1,478 | 1,229; 1,235 |
| T13 | 🟡 | AuxK targeting the whole token at $k = 1$, $\alpha$ = 1/32, 1/8, 1/2 | Subset | 1 | 1,525; 1,555; 1,555 | 1,503; 1,542; 1,538 | 1,258; 1,304; 1,302 |
| | | **$k$-annealing with more training** (Experiment 7: as T10, varying steps and learning rate) | | | | | |
| T14 | 🟡 | 4,000 steps at ½ the Gao rate | Subset | 1 | 1,479 | 1,426 (71%) | 1,187 |
| T15 | 🟡 | 4,000 steps at the Gao rate (T10 rerun) | Subset | 1 | 1,515 | 1,495 (75%) | 1,241 |
| T16 | 🟡 | 8,000 steps at the Gao rate | Subset | 1 | 1,537 | 1,529 (76%) | 1,266 |
| T17 | 🟡 | 4,000 steps at 2 times the Gao rate | Subset | 1 | 1,553 | 1,555 (78%) | 1,300 |
| T18 | 🟡 | 16,000 steps at the Gao rate | Subset | 1 | 1,589 | 1,568 (78%) | 1,333 |

In every TopK run, every token uses exactly $k$ features at the end: no token has a zero code, and the fixed-$k$ runs fill all $k$ slots. So the mean, the median and the whole range are $k$, and there is no spread to report.

No TopK run reaches 90%.

**Noise.** All runs use one seed. T15 repeats T10 with the same seed and settings, but on another machine and with the later grid code; it recovered 1,495 tokens against 1,499. The same cell run in the sandbox gave 1,531. So differences of about 35 tokens between single runs can come from the machine alone.

### 1a. Failure modes

- **Top1 fails (T1, T2)** because most features are dead at initialisation and never revive. At full scale, none of the 39,914 features dead at initialisation revived.
- **Fixed $k > 1$ fails (T3)** in the opposite way. Features stay alive, but codes use all $k$ slots, and recovery is gone by $k = 16$.
- **Annealing from a small $k_0$ fails (T4 to T6)** because too few features are alive at the start.
- **An auxiliary loss does not rescue the dead features (T12, T13).** AuxK trains dead features to reconstruct the residual. An argument predicts this should fail at $k = 1$. A dead feature aligned with the residual of a token coded at cosine $\cos\theta$ scores about $\lvert x\rvert\sin\theta$ on that token, against $\lvert x\rvert\cos\theta$ for the incumbent: at $\cos\theta = 0.9$, $0.44\lvert x\rvert$ against $0.9\lvert x\rvert$. Standard AuxK gave slightly fewer tokens than T10 without it (1,476 and 1,478, against 1,499). The whole-token variant gave up to 43 more (1,542). All runs used one seed, so we cannot tell whether a difference of this size is real. For comparison, near-identical schedules in the subset grid differ by up to about 40 tokens. Even before the final drop, AuxK could not help much, because features die *at* the drops, not between them.

### 1b. Partial successes: annealing from a large $k$

- **Surviving features are almost all correct.** In Experiment 3 (T11), 97% of the alive features are recovered tokens, with a median best cosine of 0.99997.
- **The shortfall is entirely features dying at the drops.** Alive features peak at 50,249 at $k = 128$. About 4,000 die at the drop to $k = 2$, and about 7,000 more (16%) at the drop to $k = 1$. On the subset, too, every schedule loses 15% to 20% at the final drop.
- **Dead features almost never revive.** Over the 6,400 steps at $k = 2$, the alive count rose by only 244. After the drop to $k = 1$, it rose by 3. The one exception is a large $k$ at the start: $k_0 = 128$ revived all 3,349 features dead at initialisation within 800 steps.
- **What matters most is the starting $k$ (T6 to T10).** On the subset, recovery rises with $k_0$ until nearly every feature is alive at the start. The shape of the low end changes it by at most about 40 tokens.
- **More training helps, but less with each doubling (T14 to T18).** As for ReLU, progress depends on learning rate × steps: twice the steps (T16) and twice the learning rate (T17) give similar results. Per doubling of learning rate × steps, recovery rose by 69, then about 47, then about 26 tokens, to 1,568 (78%). More training helps in two ways. Fewer features die at the drop from $k = 4$ to $k = 2$, and more features are the top feature of some token at the end of the $k = 2$ stage, which is what decides survival at the drop to $k = 1$. That number rises mostly in the first quarter of the $k = 2$ stage and then levels off, which is why a longer $k = 2$ stage alone did not help. The drop to $k = 1$ still costs 18% to 20% of the alive features in every run. Whether the gains keep shrinking is being tested with longer runs and extra seeds (the Experiment 7 extension).
- **The subset predicts full scale well.** The same schedule recovered 75% of the subset (T10) and 70% of the full table (T11).

## Part 2: ReLU SAEs with an L1 penalty

### Theory: no zero-loss dictionary, but the embeddings are still the natural target

The loss is $\lvert x - \hat{x}\rvert^2 + \lambda\sum_j f_j$, with codes $f_j \ge 0$ and $\hat{x} = \sum_j f_j d_j$. With $\lambda > 0$, no dictionary reaches zero loss. The embeddings remain the natural target, because no code can do better for a token than a single feature pointing along it, with coefficient $\lvert x\rvert - \lambda/2$. With $m = V$, every token can have its own such feature, so the normalised embeddings minimise the total loss. Experiment 2 shows exactly this shrinkage: each token's coefficient was its norm minus 0.15, which is $\lambda/2$ for $\lambda = 0.3$.

**The write-up should give the argument step by step, at the level of Appendix A below**, not in compressed form.

The argument assumes the encoder can produce exact one-hot codes: each feature must be inactive on every other token. This holds for a token exactly when its embedding is a vertex of the convex hull of the embedding table (Appendix B). For GPT-2 it holds for every token, so the minimiser is reachable in principle. The results of the check are at the end of Appendix B.

How well a token is separated predicts whether Experiment 2 recovered it. The table uses the own-direction margin of Appendix B: how far the token's own projection exceeds the largest projection of any other token, as a fraction of its norm.

| Own-direction margin | Tokens | Recovered in Experiment 2 |
|---|---|---|
| 0 or below (test fails) | 172 | 37% |
| 0 to 0.1 | 1,431 | 64% |
| 0.1 to 0.2 | 7,467 | 77% |
| 0.2 to 0.3 | 13,107 | 91% |
| 0.3 to 0.4 | 13,480 | 99.3% |
| Above 0.4 | 14,600 | 100% |

So the unrecovered 7% are not tokens that a ReLU encoder cannot isolate. They are tokens it can isolate only through a narrow gap, which training did not find.

### Empirical findings: where recovery happens

On the subset, Experiment 4 maps recovery over $\lambda$ (0.03 to 1), the learning rate (¼ to 4 times the Gao rule) and the batch size (256 to 4,096), at up to 8 million training tokens, with one seed.

| # | Outcome | $\lambda$ | Scale and training | Best cell (learning rate × Gao rate, batch size) | Recovered > 0.9 | Training needed to converge (steps at the Gao rate) |
|---|---|---|---|---|---|---|
| R1 | 🟢 | 0.03 | Subset, 8 million tokens | 1, 256 | 1,959 (98%), possibly still rising | 32,000 or more |
| R2 | 🟢 | 0.1 | Subset, 8 million tokens | 1, 1,024 | 1,899 (95%) | about 8,000 |
| R3 | 🟡 | 0.3 | Subset, 8 million tokens | 1, 256 | 1,753 (88%) | about 4,000 |
| R4 | 🟡 | 1 | Subset, 8 million tokens | ¼, 256 | 1,346 (67%) | about 2,000 |
| R5 | 🟢 | 0.3 (Experiment 2) | Full table, 85,100 steps at batch 1,024 | 1, 1,024 | **46,590 (93%)** | 85,100 run; convergence not measured |
| R10 | 🔴 | 0, no penalty (Experiment 10, control) | Subset, 8 million tokens | 1, 256 | 0 (0%) | never sparsifies |

The failing cells of the grid are not listed here. They are the two failure modes described below: too little training, and too much pressure.

Unlike TopK, a ReLU SAE lets each token use any number of features, so the codes have a spread. Active features per token, in the best cells above and in Experiment 2 at full scale:

| # | Outcome | $\lambda$ | Mean | Median | Share of tokens using 1 | 2 | 3 | 4 or more | Most used by any token |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 🟢 | 0.03 | 1.16 | 1 | 90.1% | 5.9% | 2.7% | 1.4% | 6 |
| R2 | 🟢 | 0.1 | 1.27 | 1 | 88.5% | 5.0% | 2.1% | 4.5% | 10 |
| R3 | 🟡 | 0.3 | 1.35 | 1 | 78.4% | 13.1% | 5.0% | 3.6% | 6 |
| R4 | 🟡 | 1 | 1.36 | 1 | 68.1% | 28.2% | 3.3% | 0.5% | 5 |
| R5 | 🟢 | 0.3 (Experiment 2, full table) | 1.37 | 1 | 83.2% | 8.3% | 3.0% | 5.4% | 11 |

Every token uses at least one feature. A weaker penalty leaves more tokens on a single feature. At $\lambda = 1$, codes are as sparse on average, but more tokens share two features, since fewer features are alive.

A feature counts as active on a token if its coefficient is above zero, as in Part 1 and in the logged `mean_l0`.

**A cutoff on small coefficients changes little.** One could argue that a feature with a tiny coefficient should not count as used. A natural cutoff is relative to the token's largest coefficient $f_{\max}$. This coefficient is about $\lvert x\rvert - \lambda/2$, so it depends on both the token and the run. The embedding norms $\lvert x\rvert$ range from 2.45 to 6.3, with a median of 3.9. At $\lambda = 0.3$, for example, $f_{\max}$ then ranges from about 2.3 to 6.2. A fixed absolute cutoff would therefore mean different things for different tokens and different values of $\lambda$. If the decoder rows are roughly orthogonal, a feature with coefficient $f$ contributes about $(f/f_{\max})^2$ of the reconstruction's squared norm. A cutoff at $f \ge 0.1\,f_{\max}$ therefore drops only features that contribute about 1% or less.

With this cutoff, the share of tokens using one feature changes by at most 2 percentage points in every row of the table above. The cutoff mainly trims the tail. In Experiment 2, the share using 4 or more features falls from 5.4% to 2.9%, and the mean from 1.37 to 1.27. The secondary coefficients are mostly not small: their median is 20% to 45% of the main coefficient, depending on the run.

There are three regimes:

- **Undertrained: dense codes.** At a small $\lambda$, a low learning rate or too few steps, every feature stays alive, but tokens still use many features each. This is a slow success, not a true failure. These runs are on the way to recovery and get there with more steps. An earlier sweep at a fixed 4,000 steps had wrongly classed $\lambda = 0.03$ as a failure for this reason.
- **Successful.** Across $\lambda$ from 0.03 to 0.5, at learning rates up to the Gao rule, enough training gives over 80% recovery. The best results come from the smallest $\lambda$, at the cost of the longest training.
- **Over-pressured: dead features.** A large $\lambda$ or a high learning rate kills features early. At 4 times the Gao rate with $\lambda = 1$, only 3 to 7 features survive. These deaths happen in the first half-million tokens, and more training does not undo them.

Two rules describe the successful region:

- **Training runs on learning rate × steps, not tokens seen.** Runs with the same product reach similar sparsity, whatever their batch size. Adam moves each weight by about the learning rate per step, so a larger batch gives a cleaner gradient but no faster progress.
- **A weaker penalty recovers more, but needs longer.** A larger $\lambda$ kills more features on the way to one-hot codes, so its final recovery is lower. A smaller $\lambda$ kills fewer, but sparsifies more slowly. This is the ReLU form of the trade-off in Part 1. Measured as the steps at the Gao rate until codes reach 2 active features per token on average, the time needed grows roughly as $\lambda^{-0.7}$ between $\lambda = 0.03$ and 0.3: about 22,500 steps at 0.03, 5,000 to 8,000 at 0.1, and 2,000 to 4,000 at 0.3, in the cells without deaths. Above $\lambda = 0.3$ it stops falling. (Evaluations every half million tokens limit the precision to within a factor of about 1.5.)

**The penalty does all the work (Experiment 10, `docs/experiment_10_relu_no_penalty.md`).** As $\lambda$ falls, recovery rises, so one might ask whether $\lambda = 0$ does best of all. It does not: it recovers nothing (R10). With no penalty, every exact reconstruction has zero loss, so the bound of Appendix A no longer singles out the embeddings. The run reconstructs as well as the best penalised run (relative loss about 0.001), but with dense codes of about 670 features per token. Its decoder rows stay close to random: the median best cosine between a token and any decoder row rises only from 0.12 at initialisation to 0.16. So the trend holds only for $\lambda > 0$. As $\lambda \to 0$, the training needed to sparsify diverges, and at $\lambda = 0$ itself the loss no longer prefers the embeddings at all.

At full scale, Experiment 2 (R5: $\lambda = 0.3$, batch 1,024) recovered 46,590 tokens at cosine above 0.9 (93%) and 44,337 above 0.99 (88%). 49,869 features were alive, and tokens used 1.37 active features on average (median 1, at most 11; 83% used exactly one). It needed 85,100 steps and 47 hours. Codes went from about 25,000 active features per token to 1.37 while only about 400 features died. This gradual sparsification, one token at a time, is the contrast with TopK's abrupt drops. The unrecovered 7% lie in crowded regions of the embedding space, where one feature settles between two or three similar tokens. Experiment 6 makes this precise: these are tokens a ReLU feature can isolate only through a narrow gap (see the margin table under Theory above).

## Part 3: scaling laws

How the settings that make recovery work must change as the vocabulary grows.

### 3a. The starting $k$ for TopK

At the same $k$, fewer features are alive at initialisation in a larger vocabulary. Experiment 5 (`docs/experiment_5_alive_at_init.md`) measured the share at seven sizes, as the mean of two seeds, which differ by at most 2 percentage points:

| $V$ | $k$ = 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| 2,000 | 31% | 72% | 93% | 98% | 99.8% | 100% |
| 5,000 | 29% | 68% | 90% | 96% | 98.9% | 99.8% |
| 10,000 | 26% | 65% | 88% | 95% | 98.1% | 99.6% |
| 20,000 | 25% | 60% | 84% | 92% | 96.9% | 99.1% |
| 30,000 | 23% | 57% | 81% | 90% | 95.7% | 98.5% |
| 40,000 | 22% | 54% | 79% | 88% | 94.5% | 97.9% |
| 50,257 | 21% | 52% | 77% | 87% | 93.4% | 97.3% |

Keeping the same share alive at the start therefore requires $k_0$ to grow with $V$. For 93% alive, the $k$ needed grows from 32 at $V = 2{,}000$ to 123 at $V = 50{,}257$.

The shape of this growth depends on which tokens make up each subvocabulary:

- **Random tokens: a power law in $V$.** For random subvocabularies, $k \approx 2.90\,V^{0.345}$ for 93% alive, with an exponent of 0.34 to 0.37 across target shares of 90% to 97%. This is the effect of size alone, and the clean scaling law for the write-up.
- **Most frequent tokens: faster growth at large $V$.** Our pilots used the most frequent tokens. Their embeddings are spread more evenly in direction than a random sample (mean pairwise cosine 0.236 for the 2,000 most frequent, 0.269 for a random set, 0.341 for the 2,000 rarest). Evenly spread tokens are easier to cover, because different random features win different tokens. As a frequency subvocabulary grows, it adds tokens that crowd more closely around the shared direction, so the needed $k$ rises faster than size alone would, until it meets the random curve at the full table. Frequency is only a proxy here: the variable that matters is how concentrated the embedding directions are. Over 27 sizes, a small power plus a linear term fits this curve: $k \approx 9.0\,V^{0.16} + 0.0014\,V$ for 93% alive. This curve is the relevant one for carrying a schedule from the pilot subset to the full table.

Details are in `docs/experiment_5_alive_at_init.md`.

At the two sizes trained so far, starting runs this way gave similar recovery: 68% on the subset with $k_0 = 32$ (T8) and 70% on the full table with $k_0 = 128$ (T11). Whether this holds at the sizes in between is the next experiment.

### 3b. ReLU: recovery as a function of $\lambda$ does not depend on vocabulary size

**This is a conceptual theme of the write-up and should be emphasised.** Given the same training per token, ReLU recovery depends on $\lambda$ and on how long the run trains, and hardly at all on how many tokens there are.

The questions were how the training needed for a given $\lambda$ grows with $V$, and whether the best $\lambda$ changes. Experiment 9 (`docs/experiment_9_relu_sizes.md`) trained at 5,000, 10,000 and 20,000 tokens, at batch size 256 and the Gao rate, with steps in proportion to $V$. Each $\lambda$ therefore got the same epochs' worth at every size. The 2,000-token column is from the matching Experiment 4 cells (Gao rate, batch 256), read at the nearest epochs' worth (262, 524, 1,049 and 4,194).

| # | Outcome | $\lambda$ | Epochs' worth | $V$ = 2,000 | 5,000 | 10,000 | 20,000 |
|---|---|---|---|---|---|---|---|
| R6 | 🔴 | 1 | 256 | 830 (42%) | 1,587 (32%) | 3,291 (33%) | 8,442 (42%) |
| R7 | 🟡 | 0.3 | 512 | 1,672 (84%) | 4,333 (87%) | 8,799 (88%) | 17,867 (89%) |
| R8 | 🟢 | 0.1 | 1,024 | 1,804 (90%) | 4,604 (92%) | 9,262 (93%) | 18,033 (90%) |
| R9 | 🟢 | 0.03 | 4,096 | 1,959 (98%) | 4,925 (98.5%) | | |

Recovered tokens at cosine above 0.9. The full table fits the same picture: Experiment 2 (R5, $\lambda = 0.3$, about 1,700 epochs' worth) recovered 93%.

- **The answer to the first question: the steps needed grow in proportion to $V$.** For each $\lambda$, every size ends within a few points of the others when steps scale with $V$, which at a fixed batch size means the same epochs' worth. Experiment 4 found that, at a fixed size, the batch size hardly matters once the number of steps is fixed. Together, the rule is: steps in proportion to $V$, whatever the batch size. The sparsification curves line up in epochs' worth too: at $\lambda = 0.3$, codes go from about 200 active features per token to about 1.6 over similar epochs at every size.
- **The answer to the second question: the best $\lambda$ does not change.** At every size, a smaller $\lambda$ recovers more, given its longer training, and $\lambda = 1$ fails because about half the features die early (44% to 53% alive at the end).
- **Why this matters.** The subvocabulary is a faithful small-scale model of the full problem for ReLU SAEs. The map of Experiment 4, which was cheap to compute on 2,000 tokens, predicts what happens at larger sizes once the number of steps is scaled with $V$. The cost of reaching a given recovery grows only in proportion to $V$, through the number of steps.
- **Caveat: $\lambda = 0.1$ had not converged.** Its recovery was still rising steeply at the end at every size, so its numbers are lower bounds. The 20,000-token run also lagged the smaller sizes mid-run (50% recovered at 768 epochs' worth, against 70% at 5,000 tokens). A longer run would show whether it catches up fully. We judged the picture clear enough without it. $\lambda = 0.3$ was close to its plateau, and $\lambda = 1$ is capped by deaths.

Two further points support reading the size-independence as a property of training rather than coincidence.

The geometry of the embeddings barely changes across these sizes, so any change in recovery with $V$ can be put down to training. Experiment 6 showed that recovery tracks how widely each token is separated from the others (the own-direction margin). As $V$ grows from 2,000 to 50,257, the median margin falls only from 0.36 to 0.32, and the share of narrowly separated tokens (margin below 0.2) rises only from 16% to 18%. A clear fall in recovery with $V$ would therefore come from training (steps, learning rate or deaths), not from tokens becoming harder to isolate.

The learning rate falls with size, as $1/\sqrt{m}$, so each update is smaller at larger $V$. Yet at the same epochs' worth, larger sizes take proportionally more steps. At $\lambda = 0.3$, learning rate × steps in absolute terms is about 3 times larger at 20,000 tokens than at 2,000, while recovery is the same. So across sizes, recovery tracks steps per token of vocabulary, not the absolute learning rate × steps.

## Next experiments

Experiments 7 and 9 have run (2026-10-26 to 28). Their results are in Parts 1 and 3b. Two runs are outstanding:

1. **Experiment 7 extension** (Part 1, `docs/experiment_7_topk_lr_steps.md`). Two more seeds at 8,000 and 16,000 steps, to measure the run-to-run spread, and single runs at 32,000, 64,000 and 128,000 steps, to see whether the gains from more training keep shrinking. About 6 hours on the laptop.
2. **Experiment 8: TopK annealing at intermediate sizes** (Part 3a, `docs/experiment_8_topk_sizes.md`). The 5,000, 10,000 and 20,000 most frequent tokens, starting at $k_0$ = 42, 55 and 71, which leave 93% of features alive at initialisation, with $2V$ steps so that every size gets 1,024 epochs' worth of training. It tests whether matching the share alive at the start also matches recovery at the end. About 5 hours on the laptop.

```
caffeinate -i python experiments/exp7_topk_lr_steps/run.py
caffeinate -i python experiments/exp8_topk_sizes/run.py
```

## Appendix A: why the embeddings minimise the L1 loss

The write-up should explain the Part 2 theory at this level of detail.

**Setup.** Take one token $x$. The SAE reconstructs it as $\hat{x} = \sum_j f_j d_j$, with coefficients $f_j \ge 0$ and unit-length decoder rows $\lvert d_j\rvert = 1$. Its loss is

$$L = \lvert x - \hat{x}\rvert^2 + \lambda \sum_j f_j .$$

**Step 1: the penalty is at least the length of the reconstruction.** By the triangle inequality,

$$\lvert \hat{x}\rvert = \Big\lvert \sum_j f_j d_j \Big\rvert \le \sum_j f_j \lvert d_j\rvert = \sum_j f_j .$$

So $L \ge \lvert x - \hat{x}\rvert^2 + \lambda\lvert\hat{x}\rvert$. The bound depends only on the vector $\hat{x}$, not on how it was built.

**Step 2: the best possible reconstruction.** Say $\hat{x}$ has length $t$. Among all vectors of length $t$, the one closest to $x$ points along $x$, so the error is at least $(\lvert x\rvert - t)^2$. Therefore

$$L \ge h(t) = (\lvert x\rvert - t)^2 + \lambda t .$$

The derivative is $h'(t) = -2(\lvert x\rvert - t) + \lambda$, which is zero at $t = \lvert x\rvert - \lambda/2$. There

$$h = \left(\tfrac{\lambda}{2}\right)^2 + \lambda\left(\lvert x\rvert - \tfrac{\lambda}{2}\right) = \lambda\lvert x\rvert - \tfrac{\lambda^2}{4}.$$

This is a lower bound on the token's loss, whatever the dictionary and whatever the code. It assumes $\lvert x\rvert > \lambda/2$, which holds here: the smallest GPT-2 embedding norm is 2.45.

Note that the minimum is not at full length. At $t = \lvert x\rvert$ the loss is $\lambda\lvert x\rvert$, but $h'(\lvert x\rvert) = \lambda > 0$, so shrinking the reconstruction slightly lowers it. Shortening saves penalty at rate $\lambda$, while the squared error grows from zero only quadratically. For example, with $\lvert x\rvert = 4$ and $\lambda = 0.3$:

| $t$ | Squared error | Penalty | Loss |
|---|---|---|---|
| 4 (full length) | 0 | 1.2 | 1.2 |
| 3.85 (shrunk by $\lambda/2$) | 0.0225 | 1.155 | 1.1775 |

**Step 3: one feature pointing at the token reaches the bound.** Use the single feature $d = x/\lvert x\rvert$ with coefficient $f = \lvert x\rvert - \lambda/2$. The reconstruction is $x$ shortened by $\lambda/2$, so the error is $(\lambda/2)^2$, and the penalty is $\lambda(\lvert x\rvert - \lambda/2)$. They add to $\lambda\lvert x\rvert - \lambda^2/4$, exactly the bound.

**Conclusion.** No code can do better for a token than a single feature pointing along it, with coefficient $\lvert x\rvert - \lambda/2$. With $m = V$, every token can have its own such feature, so every token sits at its lower bound at once, and the normalised embeddings minimise the total loss. The only assumption is that the ReLU encoder can output this code exactly: the token's own feature active, every other feature at zero. Appendix B turns this assumption into a geometric condition.

## Appendix B: when can a ReLU encoder output one-hot codes?

The write-up should explain this at the same level of detail as Appendix A.

**What is needed.** The encoder computes $f_j = \mathrm{ReLU}(w_j \cdot x + b_j)$, with an encoder row $w_j$ and a bias $b_j$ for each feature. Write $x_1, \dots, x_V$ for the embeddings, with feature $j$ assigned to token $j$. The code of Appendix A requires two things of feature $j$:

- **On its own token:** $w_j \cdot x_j + b_j = \lvert x_j\rvert - \lambda/2$, which is positive.
- **On every other token:** $w_j \cdot x_i + b_j \le 0$ for all $i \ne j$, so the ReLU outputs zero.

**Step 1: the condition splits into one question per token.** The encoder is not tied to the decoder, so each pair $(w_j, b_j)$ can be chosen on its own. The two requirements on feature $j$ say: the hyperplane $w_j \cdot x + b_j = 0$ has $x_j$ strictly on its positive side and every other token on or behind it.

**Step 2: the scale is free.** Suppose some pair $(w, b)$ separates $x_j$ this way, with $w \cdot x_j + b = s > 0$. Multiplying both by $(\lvert x_j\rvert - \lambda/2)/s$ keeps every other token on or behind the hyperplane and sets the coefficient on $x_j$ to exactly $\lvert x_j\rvert - \lambda/2$. So feature $j$ can produce the required code exactly when some hyperplane separates $x_j$ from the other tokens. The value of $\lambda$ plays no part.

**Step 3: separable means a vertex of the convex hull.** A point can be separated from a finite set of points exactly when it is not a weighted average (a convex combination) of them. One direction is short. If $x_j = \sum_{i \ne j} c_i x_i$ with $c_i \ge 0$ and $\sum_i c_i = 1$, then for any $w$ and $b$,

$$w \cdot x_j + b = \sum_{i \ne j} c_i\,(w \cdot x_i + b) \le 0,$$

since every term in the sum is at most zero. So the coefficient on $x_j$ cannot be positive. The other direction is the separating hyperplane theorem. A token that is not a weighted average of the others is a vertex of the convex hull of the embedding table.

**Conclusion.** One-hot codes are possible for exactly the tokens whose embeddings are vertices of the convex hull of the table. If every embedding is a vertex, the minimiser of Appendix A is reachable. A token that is not a vertex can never have a feature of its own, so it must be coded as a mixture of features, at a loss above its bound.

**How to check it.** Two tests, from cheap to exact:

1. **The own-direction test (sufficient).** Take $w_j = x_j/\lvert x_j\rvert$. This separates $x_j$ if no other token projects further along $x_j$'s direction than $x_j$ itself: $\max_{i \ne j} w_j \cdot x_i < \lvert x_j\rvert$. The bias then sits between the two values. A token that passes is a vertex.
2. **The linear-programme test (exact).** For a token that fails the first test, a different direction may still separate it. A linear programme decides whether any separating hyperplane exists: 769 unknowns ($w_j$ and $b_j$) and one constraint per other token.

The first test differs from a tied encoder. With $w_j$ tied to the decoder row $x_j/\lvert x_j\rvert$, the coefficient on $x_j$ forces $b_j = -\lambda/2$, so every other token needs $w_j \cdot x_i \le \lambda/2$. That is far stricter, and depends on $\lambda$. Our SAEs have untied encoders, so only the hull condition matters.

**Results for GPT-2 (Experiment 6, `docs/experiment_6_hull_vertices.md`).**

| Test | Full table (50,257) | Subset (2,000) |
|---|---|---|
| Pass the own-direction test | 50,085 | 1,999 |
| Fail it, but separable by linear programme | 172 | 1 |
| Not separable | 0 | 0 |

Every embedding is a vertex of the hull. For 99.66% of tokens, the token's own direction already separates it.

The linear programme bounds $w$ by $\lvert w_i\rvert \le 1$ in each coordinate and maximises $w \cdot x_j + b$. Constraints are added in rounds: the 2,000 tokens that project furthest along $x_j$'s direction come first, then any violated constraints, until none remain. The margin it reports (the gap to the nearest other token along the unit vector $w/\lvert w\rvert$, as a fraction of $\lvert x_j\rvert$) is a lower bound on the best possible margin. The 172 tokens fall into two groups:

- **115 tokens, margin 0.32 to 0.40.** They are well separated, just not along their own direction. Experiment 2 recovered 63 of them.
- **57 tokens, margin about 0.037.** 56 of them never occur in the Wikipedia count, so they are probably unused tokens. Experiment 2 recovered 1 of them.

## Appendix C: training setup (for the methodology section)

The write-up's methodology section should give a brief overview of these details. This appendix records them in one place.

### Common to every training run

- **Data.** The rows of the GPT-2 Small input embedding table, or of a subvocabulary of its $V$ most frequent tokens. Frequency (in English Wikipedia) only decides which tokens make up a subvocabulary.
- **Sampling.** Each minibatch is drawn uniformly at random from the vocabulary, with replacement. There is no frequency weighting: every token gets the same share of updates. A frequency-weighted toggle exists but was never used.
- **Steps and epochs.** A step is one optimiser update on one minibatch. With $V$ tokens and batch size $B$, one epoch's worth of tokens goes by every $V/B$ steps. Sampling is with replacement, so these are not exact passes through the vocabulary. Every run sees each token hundreds to tens of thousands of times.
- **Model.** $m = V$ features. Decoder rows of unit norm, kept so by removing the gradient component along each row before the update and renormalising after it. No decoder bias. Initialisation: random Gaussian decoder rows, the encoder a copy of the decoder, zero encoder bias. No information from the embeddings is used.
- **Optimiser.** Adam with default betas and epsilon, and a constant learning rate. The base rate ("the Gao rate") is $2\times10^{-4}/\sqrt{m/2^{14}}$, so it depends on the vocabulary size:

| $m = V$ | 2,000 | 5,000 | 10,000 | 20,000 | 50,257 |
|---|---|---|---|---|---|
| Gao rate | $5.7\times10^{-4}$ | $3.6\times10^{-4}$ | $2.6\times10^{-4}$ | $1.8\times10^{-4}$ | $1.1\times10^{-4}$ |

- **Training length** is measured in two ways. *Epochs' worth*: steps × $B$ / $V$. *Gao-rate steps*: steps × (learning rate / Gao rate). At a fixed $V$, progress depends mainly on Gao-rate steps, whatever the batch size and learning rate (Experiments 4 and 7). Across sizes, for ReLU, the steps needed grow in proportion to $V$ (Experiment 9, at a fixed batch size).
- **Evaluation.** Over every token of the vocabulary, weighted equally. A token is recovered if some decoder row is its mutual nearest neighbour at cosine above 0.9 (and 0.99). A feature is alive if it is active on at least one token. A feature is active on a token if its coefficient is positive.
- **Seeds.** One seed (0) per run, except the Experiment 7 extension (seeds 0, 1 and 2 at two lengths).
- **Hardware.** Full-scale runs and Experiments 7 to 9 ran on the laptop (CPU, PyTorch 2.2.2). Pilots ran in a cloud sandbox. The same run gave 1,495 and 1,531 recovered tokens on the two machines, so results from different machines can differ by about 35 tokens on the subset.

### Per experiment

| Experiment | Model | $V$ | Batch size | Steps | Learning rate (× Gao) | Epochs' worth | Rows in the plan |
|---|---|---|---|---|---|---|---|
| Subset pilots, fixed $k$ | TopK, fixed $k$ | 2,000 | 512 | 3,000 | 1 | 768 | T3 |
| Subset pilots, annealing and AuxK | TopK, annealed | 2,000 | 512 | 4,000 | 1 | 1,024 | T2, T4 to T10, T12, T13 |
| Subset pilot, $\lambda$ sweep | L1 ReLU | 2,000 | 1,024 | 4,000 | 1 | 2,048 | Part 2 (earlier sweep) |
| 1 | Top1 | 50,257 | 4,096 | 16,820 | 1 | 1,371 | T1 |
| 2 | L1 ReLU, $\lambda = 0.3$ | 50,257 | 1,024 | 85,100 | 1 | 1,734 | R5 |
| 3 | TopK, annealed from 128 | 50,257 | 4,096 | 15,980 | 1 | 1,302 | T11 |
| 4 | L1 ReLU, $\lambda$ from 0.03 to 1 | 2,000 | 256, 1,024, 4,096 | 32,768, 8,192, 2,048 | ¼, 1, 4 | 4,194 in every cell ($2^{23}$ tokens) | R1 to R4 |
| 5 | none (initialisation only) | 2,000 to 50,257 | | | | | Part 3a |
| 6 | none (geometry only) | 2,000 and 50,257 | | | | | Appendix B |
| 7 | TopK, annealed from 128 | 2,000 | 512 | 4,000 to 16,000 (extension: up to 128,000) | ½, 1, 2 | 1,024 to 4,096 (extension: up to 32,768) | T14 to T18 |
| 8 | TopK, annealed from 42, 55, 71 | 5,000, 10,000, 20,000 | 512 | $2V$: 10,000, 20,000, 40,000 | 1 | 1,024 at every size | Part 3a |
| 9 | L1 ReLU, $\lambda$ = 1, 0.3, 0.1 (and 0.03 at 5,000) | 5,000, 10,000, 20,000 | 256 | $V$, $2V$, $4V$ (and $16V$) | 1 | 256, 512, 1,024 (and 4,096) at every size | Part 3b |
| 10 | ReLU, $\lambda = 0$ (no penalty) | 2,000 | 256 | 32,768 | 1 | 4,194 | R10 |

Experiments 8 and 9 both hold the epochs' worth fixed across sizes, because their step counts grow in proportion to $V$. A TopK feature gets gradient only when it is selected for a token in the minibatch, so the training each feature receives scales with epochs' worth, not with steps alone.
