# Experiment 6: can a ReLU encoder output one-hot codes?

Status: done, 2026-10-26 (sandbox, 20 to 25 minutes).

## Question

The ReLU theory of Part 2 (Appendix A of `docs/write_up_plan.md`) shows that the normalised embeddings minimise the L1 loss. It assumes the encoder can output an exact one-hot code for each token: the token's own feature active, every other feature at zero. Appendix B shows that this holds for a token exactly when its embedding is a vertex of the convex hull of the embedding table. Is every GPT-2 embedding a vertex?

## Method

No training is involved. Each token is tested on its own, from cheap to exact:

1. **The own-direction test.** Take the encoder row along the token's own direction, $w_j = x_j/\lvert x_j\rvert$. The token passes if no other token projects further along $w_j$ than $x_j$ itself. Its **own-direction margin** is the gap, as a fraction of $\lvert x_j\rvert$:
   $$\frac{\lvert x_j\rvert - \max_{i \ne j} w_j \cdot x_i}{\lvert x_j\rvert}.$$
   A token with a positive margin is a vertex.
2. **A linear programme**, for the tokens that fail the first test. It maximises $w \cdot x_j + b$ subject to $w \cdot x_i + b \le 0$ for every other token, with $-1 \le w_i \le 1$ in each coordinate. The token is a vertex exactly when the optimum is positive. Constraints are added in rounds: first the 2,000 tokens that project furthest along $x_j$'s direction, then up to 2,000 of the most violated constraints per round, until none is violated. The reported margin is the gap to the nearest other token along $w/\lvert w\rvert$, as a fraction of $\lvert x_j\rvert$. It is a lower bound on the best possible margin.

Finally, Experiment 2's recovery (cosine above 0.9) is tabulated against the own-direction margin.

## Results

| Test | Full table (50,257) | Subset (2,000) |
|---|---|---|
| Pass the own-direction test | 50,085 | 1,999 |
| Fail it, but separable by linear programme | 172 | 1 |
| Not separable | 0 | 0 |

**Every embedding is a vertex of the hull**, so exact one-hot codes exist for every token. No two embeddings are identical. For 99.66% of tokens, the token's own direction already separates it.

The tokens that fail the own-direction test have shorter embeddings than average (median norm 3.25, against 3.95 for the table). They fall into two groups:

- **115 tokens, margin 0.13 to 0.40 (median 0.34).** They are well separated, just not along their own direction. Experiment 2 recovered 63 of them.
- **57 tokens, margin below 0.1** (all but one between 0.037 and 0.042). 56 of them never occur in the Wikipedia count, so they are probably unused tokens. Experiment 2 recovered 1 of them.

**The margin predicts recovery in Experiment 2.**

| Own-direction margin | Tokens | Recovered in Experiment 2 |
|---|---|---|
| 0 or below (test fails) | 172 | 37% |
| 0 to 0.1 | 1,431 | 64% |
| 0.1 to 0.2 | 7,467 | 77% |
| 0.2 to 0.3 | 13,107 | 91% |
| 0.3 to 0.4 | 13,480 | 99.3% |
| Above 0.4 | 14,600 | 100% |

## Reading

- The minimiser of the ReLU loss can be reached in principle for every token. The theory of Part 2 needs no exceptions.
- The unrecovered 7% of Experiment 2 are not tokens that a ReLU encoder cannot isolate. They are tokens it can isolate only through a narrow gap. A feature's bias must sit inside that gap, and training did not find it.

## How to run

```
python experiments/exp6_hull_vertices/run.py
python experiments/exp6_hull_vertices/run.py --subset 2000
```

Results go to `runs/exp6/hull_full.csv` (or `hull_subset2000.csv`), one row per token: its Wikipedia count, own-direction margin, the linear-programme optimum and margin (for tokens that fail the first test), and whether it is a vertex. If `runs/exp2/model.pt` exists (or `--exp2-run` points to it), the recovery table above is printed. The full table takes 20 to 25 minutes in the sandbox, almost all in the 172 linear programmes, with a progress line and ETA every minute.
