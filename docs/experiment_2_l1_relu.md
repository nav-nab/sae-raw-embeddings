# Experiment 2: an L1-penalised ReLU SAE

Status: ready to run, with $\lambda = 0.3$.

This experiment runs if fewer than 80% of the Top1 features that are dead at initialisation revive in Experiment 1. None revived (`docs/experiment_1_results.md`), so it goes ahead.

## Motivation

In the Top1 SAE, only the winning feature receives gradient. A feature that never wins is frozen, and most features never win from a random start. An L1-penalised ReLU SAE removes this winner-take-all mechanism. Every feature whose pre-activation is positive on a token receives gradient from that token. Because the embeddings share a large mean component, most random features should be active on at least some tokens at initialisation. The dead-feature problem should therefore be much milder.

The change has a price. The clean target of the Top1 setting no longer holds exactly. The normalised embeddings are still the best dictionary among those that reconstruct perfectly. With unit-norm atoms, the L1 norm of a perfect code is at least $\|x\|_2$. Equality holds only when a single atom points along $x$. The trouble is the encoder. A ReLU encoder with one bias per feature cannot produce the exact one-hot codes. Some tokens have a neighbour at cosine 0.998, and token norms range from 2.4 to 6.3. A threshold high enough to silence a neighbour also shrinks the correct coefficient. The L1 penalty adds further shrinkage of about $\lambda/2$. So zero loss is out of reach. Success has to be judged by whether the decoder recovers the embedding directions, not by the loss.

## Question

Does an L1-penalised ReLU SAE, trained from random initialisation, recover the token embedding directions in its decoder?

## Design

Shared with Experiment 1:

- **Data:** the GPT-2 Small input embedding table, with $V = 50{,}257$ rows and $d = 768$. Tokens are sampled uniformly. Frequency weighting is off.
- **Dictionary:** $m = V$ features, unit-norm decoder rows, zero decoder bias.
- **Initialisation:** random unit-norm Gaussian decoder rows. The encoder is a copy of the decoder, and the encoder bias is zero. The initial parameters are saved to `init.pt`.
- **Training:** Adam with default betas and epsilon, one seed, regular full-vocabulary evaluations, and a budget of about 10 hours on the laptop.

Different from Experiment 1:

- **Encoder:** $f(x) = \mathrm{ReLU}(W_{\mathrm{enc}} x + b_{\mathrm{enc}})$. Every feature may be active.
- **Loss:** $\|x - W_{\mathrm{dec}}^\top f(x)\|_2^2 + \lambda \sum_j f_j(x)$. With unit-norm decoder rows, this is the usual SAE L1 penalty.
- **Penalty:** $\lambda = 0.3$, chosen by the subset sweep below.
- **Batch size:** 1024, since the dense decoder makes large batches slow.
- **Cost:** in the sandbox at full scale, a step at batch size 1024 took 2.6 to 3.3 s and a full-vocabulary evaluation about 55 s, twice as long as for Top1. The final analysis took about 2 minutes.

## Measurement

- **Logged at each evaluation:** loss, relative loss, the mean number of active features per token, the number of alive features, and revival of features dead at initialisation. A feature is alive if it is active on at least one token.
- **Recovery:** the number of embeddings that have a decoder row at high cosine, and how many of those matches are one-to-one. A full assignment over a $50{,}257 \times 50{,}257$ matrix is too large, so we use nearest-neighbour matches in both directions.

## Choosing λ: subset sweep

These runs used the 2,000 most frequent tokens with $m = 2{,}000$, 4,000 steps at batch size 1024, the Gao et al. learning rate and seed 0. A token is recovered if it has a mutual-nearest decoder row above the given cosine.

| $\lambda$ | Alive at end (of 2,000) | Active features per token | Relative loss | Recovered at cosine > 0.9 | Recovered at cosine > 0.99 |
|---|---|---|---|---|---|
| 0.03 | 2,000 | 79.5 | 0.005 | 3 | 0 |
| 0.1 | 2,000 | 5.0 | 0.007 | 1,055 | 622 |
| 0.3 | 1,894 | 1.7 | 0.010 | 1,710 | 1,516 |
| 1 | 895 | 1.7 | 0.217 | 707 | 425 |

$\lambda = 3$ and 10 were planned but not run. At $\lambda = 1$ more than half the features had already died within the first 500 steps, and larger values would only kill more.

$\lambda = 0.3$ did best. It recovered 1,710 tokens at cosine above 0.9 and 1,516 above 0.99, with about 95% of features alive and a relative loss of 0.010. That is more than the best $k$-annealing schedule on the same subset (1,362 and 1,068).

Two trends set the window for $\lambda$. Below 0.3, codes stay dense for longer: at $\lambda = 0.1$ active features per token were still falling at the end (from 8 to 5 over the last 500 steps), and recovery was still rising. Above 0.3, features die early: at $\lambda = 1$, alive features fell from 2,000 to about 1,040 by step 500. Token norms are the same on the subset and the full table (about 4 on average), so the scale of $\lambda$ should carry over.

## How to run

```
caffeinate -i python experiments/exp2_l1_relu/run.py
```

The script works like the other experiment scripts. It prepares the data if needed and times this machine to fit the 10-hour budget. It then trains with a checkpoint every 10 minutes and a heartbeat every minute, and finishes with the analysis. Ctrl-C saves and stops, and the same command resumes. Results go to `runs/exp2/`. `--l1-coef` changes $\lambda$.

One thing to watch at full scale. At initialisation about half of all features are active on each token, about 25,000 of 50,257 against about 1,000 of 2,000 on the subset. The relative loss therefore starts near 1,000 rather than near 1, and the first part of training is spent switching most features off. On the subset this took a few hundred steps. At full scale the learning rate is five times smaller and there are 25 times more active features to silence, so it may take a good share of the run. The heartbeat's active-features-per-token figure shows how fast this is going.

## Decisions

- **Learning rate:** the same Gao et al. rule as the other experiments, about $1.1\times10^{-4}$ at full scale, so that only the encoder and loss change.
- **Recovery threshold:** the same mutual-nearest-neighbour test at cosine 0.9 and 0.99 as the other experiments.
- **Batch size:** 1024. At 4096 a step takes about three times as long, so 10 hours would allow only about 3,500 steps. At 1024 about 10,700 steps fit, for roughly the same number of tokens seen.
- **$\lambda = 0.3$**, the best value in the subset sweep above.
