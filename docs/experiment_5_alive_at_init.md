# Experiment 5: features alive at initialisation, by vocabulary size

Status: done, 2026-10-26 (sandbox, about 12 minutes in all).

## Question

For $k$-annealing, recovery depends on the share of features alive at the start (Part 1b of `docs/write_up_plan.md`). How does that share at initialisation depend on the vocabulary size $V$ and on $k$? This is the base of Part 3a: the starting $k$ needed for a given share alive, as $V$ grows.

## Design

- **Vocabularies, by frequency.** 27 sizes from 2,000 to 50,257: 2,000, 2,500, 3,000, 3,500, 4,000, 4,500, 5,000, 6,000, 7,000, 7,500, 8,500, 10,000, 12,500, 15,000, 17,500, 20,000, 22,500, 25,000, 27,500, 30,000, 32,500, 35,000, 37,500, 40,000, 42,500, 45,000 and 50,257 (the full table). Each is the $V$ GPT-2 tokens most frequent in English Wikipedia, so the subvocabularies are nested, as for the 2,000-token subset used in the pilots.
- **Vocabularies, at random (control).** 7 sizes: 2,000, 5,000, 10,000, 20,000, 30,000, 40,000 and 50,257. Each is the first $V$ tokens of one fixed random ordering of the vocabulary, so these are nested too. Random subsets keep the same embedding geometry at every size, so this separates the effect of size from the effect of a changing geometry.
- **Model.** The randomly initialised TopK SAE with $m = V$, exactly as in training: random unit-norm Gaussian decoder rows, the encoder a copy of the decoder, zero encoder bias.
- **Measurement.** A feature is alive if, for at least one token, it is among the $k$ largest pre-activations and its pre-activation is positive. Every $k$ from 1 to 256 is measured. No training is involved.
- **Seeds.** Two, 0 and 1. All tables report their mean. The two differ by at most 2 percentage points (at $V = 2{,}000$, $k = 8$) and by under 0.5 points at full scale.

## Results: share alive at initialisation

Most frequent tokens, a selection of sizes and $k$ (%):

| $V$ | $k$ = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| 2,000 | 31.1 | 43.9 | 57.7 | 72.2 | 84.0 | 93.3 | 97.9 | 99.8 | 100.0 |
| 5,000 | 28.9 | 40.2 | 54.3 | 68.3 | 80.5 | 90.1 | 96.2 | 98.9 | 99.8 |
| 10,000 | 26.4 | 37.3 | 50.6 | 64.5 | 77.3 | 87.5 | 94.5 | 98.1 | 99.6 |
| 20,000 | 24.5 | 34.7 | 46.9 | 60.3 | 73.1 | 83.8 | 92.1 | 96.9 | 99.1 |
| 30,000 | 23.0 | 32.4 | 44.1 | 56.9 | 69.8 | 81.2 | 90.0 | 95.7 | 98.5 |
| 40,000 | 21.6 | 30.7 | 41.6 | 54.2 | 67.0 | 78.8 | 88.2 | 94.5 | 97.9 |
| 50,257 | 20.5 | 29.0 | 39.8 | 52.0 | 64.7 | 76.7 | 86.5 | 93.4 | 97.3 |

At $k = 32$, random subsets have fewer features alive than frequency subsets of the same size: 91.3% against 93.3% at 2,000, 84.6% against 87.5% at 10,000, 79.2% against 81.2% at 30,000. The two converge to the same value at the full table. The most frequent tokens are easier to cover than a random sample, for the geometric reason set out below.

## Results: the starting $k$ needed

The smallest $k$ at which the mean share alive reaches 93%:

| $V$ | 2,000 | 5,000 | 10,000 | 20,000 | 30,000 | 40,000 | 50,257 |
|---|---|---|---|---|---|---|---|
| Most frequent tokens | 32 | 42 | 55 | 71 | 88 | 105 | 123 |
| Random tokens | 40 | 55 | 69 | 86 | 100 | 113 | 123 |

The full frequency series at 93%, over all 27 sizes in the order listed above, is 32, 36, 38, 38, 39, 41, 42, 44, 48, 49, 51, 55, 58, 63, 66, 71, 75, 79, 84, 88, 92, 96, 100, 105, 110, 114, 123.

## Fits

Power laws $k \approx c\,V^p$, by least squares on relative error.

**Random tokens (7 sizes).** For 93% alive, $k \approx 2.90\,V^{0.345}$, with an RMS error of 1.3%. The exponent rises slightly with the target: 0.340 for 90%, 0.355 for 95% and 0.374 for 97%.

**Most frequent tokens (27 sizes).** For 93% alive, $k \approx 1.37\,V^{0.405}$. The fits for each target:

| Target | $c$ | $p$ | RMS error | Largest error | Fit at 50,257 (measured) |
|---|---|---|---|---|---|
| 90% | 1.23 | 0.384 | 5.4% | 10.6% | 79 (88) |
| 93% | 1.37 | 0.405 | 5.5% | 11.4% | 109 (123) |
| 95% | 1.50 | 0.421 | 5.2% | 10.6% | 144 (161) |
| 97% | 1.60 | 0.451 | 5.2% | 10.3% | 212 (236) |

The power law fits the frequency subsets less tightly than the random ones. It runs low at both ends of the range (for 93%: 29.6 against 32 at 2,000 tokens, 109 against 123 at the full table) and is within about 6% from 5,000 to 20,000 tokens. The reason is the change in geometry described next.

## Why the token order matters: geometry, not frequency

The corpus plays no part in the measurement itself. Frequency only decides which tokens make up each frequency subvocabulary. It matters because the embeddings of frequent and rare GPT-2 tokens have different geometry. Unit-normalised embeddings, in blocks of 2,000 tokens by frequency rank:

| Tokens | Mean cosine between pairs | Mean cosine with the common mean direction |
|---|---|---|
| Frequency ranks 1 to 2,000 | 0.236 | 0.43 |
| Ranks 10,000 to 12,000 | 0.249 | 0.50 |
| Ranks 20,000 to 22,000 | 0.267 | 0.52 |
| Ranks 30,000 to 32,000 | 0.287 | 0.54 |
| Ranks 40,000 to 42,000 | 0.309 | 0.55 |
| Ranks 48,257 to 50,257 (rarest) | 0.341 | 0.56 |
| Random 2,000 or 10,000 tokens | 0.269 | 0.52 |

The rarer a token, the more its embedding points along one shared direction. Frequent tokens are spread out more evenly.

This geometry sets the share alive at initialisation. A random feature is alive if it is among the top $k$ for some token. When tokens point in many different directions, different features win different tokens, and many features are alive. When tokens crowd around one direction, the few features that happen to lie near it win almost every token, and fewer are alive. Experiment 1 showed the extreme case: the few surviving Top1 features pointed along the mean direction. So at the same $V$, frequent tokens are easier to cover than a random sample, and a random sample is easier than rare tokens.

## Reading

- **The effect of size alone is a power law, $k \propto V^{0.34}$ to $V^{0.37}$.** This is what the random subsets measure. The constant and the exponent go together, so the exponent should be quoted to three figures when the constant is used: rounding 0.345 to 0.34 makes the curve about 6% low at full scale.
- **The frequency subsets grow faster at large $V$ because of a change in geometry, not size.** Random subsets keep the same geometry at every size (mean pairwise cosine 0.269 at both 2,000 and 10,000 tokens), so only size changes. Frequency subsets start with the most spread-out tokens, so they need a smaller $k$ than random subsets at small $V$. Each larger frequency subset then adds tokens that crowd more closely around the shared direction, so the needed $k$ rises faster than size alone would, until the two curves meet at the full table. Their power law therefore has a larger exponent (0.405 against 0.345), and fits less tightly.
- **Frequency is only a proxy.** The variable that matters is how concentrated the embedding directions are, which in GPT-2 happens to vary with token frequency. A scaling law in $V$ should hold the geometry fixed, as random subsets do.
- **Which curve the write-up uses depends on the claim.** For a scaling law in vocabulary size, the random-subset power law is the clean one. For carrying a schedule from the pilot subset (the 2,000 most frequent tokens) to the full table, the frequency curve is the relevant one, since that is the path our experiments took.
- **This is a start-of-training measurement.** Whether matching the share alive at the start also matches recovery at the end is tested by the next experiment on the list: annealing runs at intermediate sizes.

## How to run

```
python experiments/exp5_alive_at_init/run.py
python experiments/exp5_alive_at_init/run.py --order random --sizes 2000 5000 10000 20000 30000 40000 50257
```

Results go to `runs/exp5/alive_at_init_freq.csv` and `runs/exp5/alive_at_init_random.csv`, one row per $V$, seed and $k$. The script prints the smallest $k$ reaching each target share.
