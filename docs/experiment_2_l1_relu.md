# Experiment 2: an L1-penalised ReLU SAE

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
- **Cost:** the decoder is now dense. In the sandbox a step at batch size 4096 took about 3 times as long as a Top1 step (9.9 s against 3.1 s), and needed about 4 GB of memory. At batch size 1024 it took 3.3 s.

## Measurement

- **Logged at each evaluation:** loss, relative loss, the mean number of active features per token, the number of alive features, and revival of features dead at initialisation. A feature is alive if it is active on at least one token.
- **Recovery:** the number of embeddings that have a decoder row at high cosine, and how many of those matches are one-to-one. A full assignment over a $50{,}257 \times 50{,}257$ matrix is too large, so we use nearest-neighbour matches in both directions.

## How to run

A single script, `experiments/exp2_l1_relu/run.py`, will run this experiment through the same shared runner as Experiments 1 and 3. It will be written once the design questions below are settled.

## Open design questions

1. The value of $\lambda$.
2. The learning rate. The current rule was derived for TopK SAEs.
3. The cosine threshold for counting a token as recovered.
