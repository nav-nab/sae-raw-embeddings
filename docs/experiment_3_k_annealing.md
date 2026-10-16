# Experiment 3: TopK with k-annealing

Status: planning. The schedule is to be decided.

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

## Design

The same as Experiment 1, except for the encoder and the schedule:

- **Data:** the GPT-2 Small input embedding table, with $V = 50{,}257$ rows and $d = 768$. Tokens are sampled uniformly.
- **Model:** a TopK SAE with $m = V$ features. The encoder selects the $k$ largest pre-activations and applies a ReLU to them, so a selected feature with a pre-activation at or below zero is inactive. Decoder rows have unit norm, and the decoder bias is zero.
- **Loss:** plain MSE.
- **Initialisation:** as in Experiment 1, with the same seed. The initial parameters are saved to `init.pt`.
- **Training:** Adam with the Gao et al. learning rate, batch size 4096, one seed, a budget of about 10 hours on the laptop.
- **Schedule:** $k$ starts at $k_0$ and decreases to 1 by some point in training, then stays at 1. The measurements above suggest $k_0 = 16$, which keeps about three quarters of features alive on the subset. To be decided: $k_0$, the shape of the decrease, and when it reaches 1.
- **Measurement:** as in Experiment 1. Alive features and active features per token are logged throughout. Revival, loss and decoder recovery are measured at the end, once $k = 1$.

## Decisions

- Fixed-$k$ TopK dropped in favour of $k$-annealing, based on the measurements above.
- To be decided: the schedule.

## Code

`experiments/exp3_k_annealing/` (script to be written). The model is `TopKSAE` in `sae_raw/model.py`, which is written and tested. With $k = 1$ it matches `Top1SAE`.
