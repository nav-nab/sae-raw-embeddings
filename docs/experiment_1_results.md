# Experiment 1: results

Plan: `docs/experiment_1_top1.md`. Code: `experiments/exp1_top1/run.py`. Run: 2026-10-15 to 16 on the laptop.

## Design in brief

We trained a Top1 SAE ($k = 1$) on the GPT-2 Small token embedding table. The table has $V = 50{,}257$ rows in $d = 768$ dimensions, and the dictionary has $m = V$ features. Decoder rows have unit norm and there is no decoder bias, so the exact zero-loss dictionary is the set of normalised embeddings.

The initialisation used no information from the data. Decoder rows were random unit-norm Gaussian directions, the encoder was a copy of the decoder, and the encoder bias was zero. Tokens were sampled uniformly. Training used Adam at a learning rate of $1.14\times10^{-4}$ (the Gao et al. rule) with batch size 4096. It ran for 16,820 steps, or 7.8 hours, with a full-vocabulary evaluation every 841 steps.

The question was what share of the features dead at initialisation would revive during training. A feature is dead if it wins no token. A dead feature gets no gradient, so it revives only if its parameters change. The plan set 80% as the level needed to stay with Top1.

## Results

**No dead feature revived.** At initialisation, 39,914 of the 50,257 features were dead. None of them revived.

**The alive features collapsed as well.** 10,343 features were alive at step 0. By the first evaluation, at step 841, only 246 were left. The count then settled at 259 for the rest of the run. So about 99.5% of the dictionary ended up unused.

| Step | Alive features | Relative loss |
|---|---|---|
| 0 | 10,343 | 0.977 |
| 841 | 246 | 0.716 |
| 4,205 | 259 | 0.661 |
| 16,820 (end) | 259 | 0.654 |

Relative loss is the squared reconstruction error divided by the squared norm, averaged over all tokens. A value of 1 means no better than reconstructing zero. The loss fell quickly at first and was nearly flat after the first quarter of training.

**The loss is about what 259 features can achieve.** For comparison, a single direction gives 0.728. A k-means-style fit with 259 directions, started from tokens, gives 0.639. The trained SAE gives 0.654. A k-means-style fit with 10,343 directions gives 0.386, and the exact solution gives 0. So the training made good use of the features it kept. The loss is high because almost all features are unused.

**Almost no embeddings are recovered.** Only 16 tokens have a matching feature (a mutual nearest neighbour) at cosine above 0.9, and 6 above 0.99. These matches are all tiny features that each win a single token, such as `ran` and `iter`.

## What the surviving features represent

The surviving features are very unequal in size. The largest wins 3,040 tokens (6% of the vocabulary). The 10 largest win 29% and the 50 largest win 75%. The median feature wins 37 tokens.

The large features point almost along the mean embedding direction (cosine 0.90 to 0.95). Their nearest tokens by cosine are GPT-2's undertrained "glitch" tokens, such as ` externalToEVA`, ` TheNitrome` and control characters, which also lie near the mean. Those nearest tokens say little about the feature.

The tokens each feature wins are more telling, and they are grouped by surface form rather than meaning:

| Feature | Tokens won | Sample of the tokens it wins |
|---|---|---|
| 362 | 3,040 | `roying`, `ribute`, `ruption`, `emetery` (lowercase pieces, no leading space) |
| 8093 | 1,489 | ` Julia`, ` Alberto`, ` Monica`, ` Zuckerberg` (capitalised first names) |
| 41193 | 1,374 | ` Tu`, ` Ton`, ` Patt`, ` Sar` (short capitalised fragments) |
| 30775 | 1,371 | ` Participant`, ` Organisation`, ` Transmission` (capitalised whole words) |
| 18838 | 37 | `uting`, `uted`, `ution`, `utive` (pieces starting with `u`) |
| 40750 | 9 | `ground`, ` Ground`, ` grounds`, ` grounding` (one word family) |

The clusters follow the leading space, capitalisation and first letters of tokens. Only the small features show a clean group of related words, such as the `ground` family.

## Interpretation

We read this as a failure of the Top1 training regime. With $k = 1$, only the winning feature for each token receives gradient. The cutoff is too harsh for features to stay alive. Most features never win at initialisation, partly because the embeddings share a large mean component. Of those that do win, a few quickly take over large groups of tokens, and the rest lose all their tokens and die.

Gradient descent therefore cannot reach the known global minimum in this setting. That minimum needs all 50,257 features, and training ends up using 259.

## Next step

Experiment 2 (`docs/experiment_2_l1_relu.md`) replaces the Top1 encoder with an L1-penalised ReLU encoder. In that model, every feature that is active on a token receives gradient from it.
