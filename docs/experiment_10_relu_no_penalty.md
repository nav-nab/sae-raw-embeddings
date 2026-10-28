# Experiment 10: a ReLU SAE with no sparsity penalty (control)

Status: done, 2026-10-28 (sandbox, 17 minutes).

## Question

Recovery in the ReLU grid rose as $\lambda$ fell, with $\lambda = 0.03$ the best so far. Does recovery survive at $\lambda = 0$, with no sparsity penalty at all?

Theory predicts not. Appendix A of `docs/write_up_plan.md` singles out the embeddings through the penalty: with $\lambda > 0$, only a single feature pointing along a token reaches the lowest possible loss, $\lambda\lvert x\rvert - \lambda^2/4$. At $\lambda = 0$ that bound is zero, and every code that reconstructs a token exactly reaches it. With $m = V$ features in 768 dimensions there are many such codes, so the loss no longer prefers the embeddings.

## Design

As the best cell of Experiment 4 (`lam0.03_lr1_b256`), except for the penalty: the 2,000 most frequent tokens, $m = 2{,}000$, batch size 256, the Gao rate, seed 0, 8 million ($2^{23}$) training tokens, which is 32,768 steps or about 4,200 epochs' worth. Evaluations every half million tokens, with decoder recovery and the spread of active features per token.

## Results

| Step | Relative loss | Active features per token (mean) | Median best cosine of a token to a decoder row | Recovered > 0.9 |
|---|---|---|---|---|
| 0 | 1.32 | 987 | 0.12 | 0 |
| 2,048 | 0.0003 | 833 | 0.14 | 0 |
| 8,192 | 0.0010 | 756 | 0.15 | 0 |
| 16,384 | 0.0010 | 720 | 0.15 | 0 |
| 32,768 | 0.0010 | 668 | 0.16 | 0 |

All 2,000 features stay alive throughout. At the end, the number of active features per token ranges up to 780, with a median of 671.

### Findings

1. **No recovery at all.** Not a single token has a decoder row within cosine 0.9 of it, at any point. The median best cosine between a token and any decoder row rises only from 0.12 at initialisation to 0.16. The decoder rows stay essentially random directions.
2. **Reconstruction is as good as in the best penalised run.** The relative loss falls to about 0.001 within the first 2,000 steps and stays there, between 0.0003 and 0.0013 from one evaluation to the next. The $\lambda = 0.03$ cell reached 0.0008 after the same training. The model solves the reconstruction problem just as well, with dense codes.
3. **Codes stay dense.** Tokens use about 670 features each at the end. The count drifts down slowly (from 833 at step 2,048 to 668 at step 32,768), but nowhere near one feature per token.

## Reading

The sparsity penalty does all the work. With the same architecture, initialisation and training, removing it removes recovery entirely. This fits the theory: at $\lambda = 0$ every exact reconstruction is a minimiser, and gradient descent settles on a dense one near its random start.

So the trend that smaller $\lambda$ recovers more holds only for $\lambda > 0$, and only given enough training. In Experiment 4, the training needed to sparsify grew as $\lambda$ fell, roughly as $\lambda^{-0.7}$ for $\lambda$ from 0.03 to 0.3 (see Part 2 of the plan). That time diverges as $\lambda \to 0$, and at $\lambda = 0$ itself the loss no longer prefers the embeddings at all.

## How to run

```
python experiments/exp10_relu_no_penalty/run.py
```

Results go to `runs/exp10/`: the run's folder (`eval_log.csv`, `result.json` and the final `model.pt`) and `summary.csv`. About 17 minutes in the sandbox. Ctrl-C saves and stops, and the same command resumes.
