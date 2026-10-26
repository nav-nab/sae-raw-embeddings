# Experiment 5: features alive at initialisation, by vocabulary size

Status: done, 2026-10-26 (sandbox, under 3 minutes).

## Question

For $k$-annealing, recovery depends on the share of features alive at the start (Part 1b of `docs/write_up_plan.md`). How does that share at initialisation depend on the vocabulary size $V$ and on $k$? This is the base of Part 3a: the starting $k$ needed for a given share alive, as $V$ grows.

## Design

- **Vocabularies.** $V$ = 2,000, 5,000, 10,000, 20,000, 30,000, 40,000 and 50,257 (the full table). Each is the $V$ GPT-2 tokens most frequent in English Wikipedia, so the subvocabularies are nested, as for the 2,000-token subset used in the pilots.
- **Model.** The randomly initialised TopK SAE with $m = V$, exactly as in training: random unit-norm Gaussian decoder rows, the encoder a copy of the decoder, zero encoder bias.
- **Measurement.** A feature is alive if, for at least one token, it is among the $k$ largest pre-activations and its pre-activation is positive. $k$ = 1, 2, 4, …, 256. No training is involved.
- **Seeds.** Two, 0 and 1. The table reports their mean.

## Results

Share of features alive at initialisation (%), mean of two seeds:

| $V$ | $k$ = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| 2,000 | 31.1 | 43.9 | 57.7 | 72.2 | 84.0 | 93.3 | 97.9 | 99.8 | 100.0 |
| 5,000 | 28.9 | 40.2 | 54.3 | 68.3 | 80.5 | 90.1 | 96.2 | 98.9 | 99.8 |
| 10,000 | 26.4 | 37.3 | 50.6 | 64.5 | 77.3 | 87.5 | 94.5 | 98.1 | 99.6 |
| 20,000 | 24.5 | 34.7 | 46.9 | 60.3 | 73.1 | 83.8 | 92.1 | 96.9 | 99.1 |
| 30,000 | 23.0 | 32.4 | 44.1 | 56.9 | 69.8 | 81.2 | 90.0 | 95.7 | 98.5 |
| 40,000 | 21.6 | 30.7 | 41.6 | 54.2 | 67.0 | 78.8 | 88.2 | 94.5 | 97.9 |
| 50,257 | 20.5 | 29.0 | 39.8 | 52.0 | 64.7 | 76.7 | 86.5 | 93.4 | 97.3 |

The two seeds differ by at most 2 percentage points (at $V = 2{,}000$, $k = 8$), and by under 0.5 points at full scale.

**The $k$ needed for a given share alive.** Interpolated between the measured powers of two (linearly in $\log_2 k$), so these values are approximate:

| $V$ | 2,000 | 5,000 | 10,000 | 20,000 | 30,000 | 40,000 | 50,257 |
|---|---|---|---|---|---|---|---|
| $k$ for 90% alive | 25 | 32 | 41 | 54 | 64 | 78 | 91 |
| $k$ for 93% alive | 31 | 45 | 55 | 73 | 92 | 108 | 123 |
| $k$ for 97% alive | 56 | 79 | 104 | 132 | 177 | 214 | 242 |

## Reading

- **At every $k$, the share alive falls steadily as $V$ grows.** For example, at $k = 32$ it falls from 93% to 77%, and the share dead rises from 7% to 23%.
- **The $k$ needed for a fixed share grows roughly fourfold** while $V$ grows 25-fold, from 2,000 to 50,257. Over the whole range this is close to $k \propto V^{0.43}$. The growth is slower at small $V$ and faster at large $V$, so a single power law fits only roughly.
- **Size and token mix are confounded.** A larger subvocabulary adds rarer tokens, so the measurement mixes the effect of $V$ with any change in the embeddings' geometry. Random subsets of each size would separate the two, if that ever matters.
- **This is a start-of-training measurement.** Whether matching the share alive at the start also matches recovery at the end is tested by the next experiment on the list: annealing runs at intermediate sizes.

## Shape of the growth (2026-10-26)

**Exact thresholds.** The smallest integer $k$ at which the mean share alive reaches each target, from the shares at every $k$ from 1 to 256:

| $V$ | 2,000 | 5,000 | 10,000 | 20,000 | 30,000 | 40,000 | 50,257 |
|---|---|---|---|---|---|---|---|
| $k$ for 90% alive | 25 | 32 | 40 | 52 | 65 | 76 | 88 |
| $k$ for 93% alive | 32 | 42 | 55 | 71 | 88 | 105 | 123 |
| $k$ for 95% alive | 40 | 53 | 70 | 92 | 115 | 138 | 161 |
| $k$ for 97% alive | 51 | 73 | 100 | 131 | 164 | 204 | 236 |

These supersede the interpolated values above.

**Fits.** Least squares on relative error, for the 93% row. The other targets give the same ranking.

| Form | Parameters | RMS relative error |
|---|---|---|
| $c\,V^p$ | 2 | 5.9% |
| $a + bV$ | 2 | 5.2% |
| $a + c\,V^p$ | 3 | 2.1% |
| $a\,V^p + bV$ | 3 | 1.5% |

- **Above about 20,000 tokens the growth is close to linear in $V$:** each extra 10,000 tokens adds about 17 to $k$ for 93% alive. A single power $c\,V^p$ cannot bend this way: its slope falls as $V$ grows.
- **The best two-term form is a small power plus a linear term,** $k \approx 6.6\,V^{0.20} + 0.0013\,V$ for 93%. The power term dominates at small $V$ and the linear term at large $V$.
- **Caveats.** Seven points support three parameters only loosely, so the forms should not be extrapolated far. Also, larger subvocabularies add rarer tokens, so the linear growth may partly reflect the token mix rather than size alone.

## How to run

```
python experiments/exp5_alive_at_init/run.py
```

Results go to `runs/exp5/alive_at_init.csv` (one row per $V$, seed and $k$), and the mean table is printed.
