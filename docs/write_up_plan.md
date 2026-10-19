# Write-up plan

Status: framing agreed; more exploratory work on the subset comes before the write-up.

## Thesis

With a dictionary of $m = V$ features trained on the raw token embedding table, theory says the SAE's features should be the token embeddings. For a Top1 SAE with unit-norm decoder rows and no decoder bias, the normalised embeddings are the zero-loss dictionary. For other sparse SAEs they remain the natural target.

The paper shows that whether training actually finds this dictionary depends heavily on the architecture and the training hyperparameters. Recovery is possible, but it is the exception. The setup that succeeded was found by trial and error.

## Part 1: the success mode

The full GPT-2 Small run of Experiment 2: an L1-penalised ReLU SAE with $\lambda = 0.3$, batch size 1024 and the Gao et al. learning rate, trained for 85,100 steps from random initialisation.

- **Result.** 46,590 of 50,257 tokens recovered at cosine above 0.9 (92.7%), and 44,337 above 0.99 (88.2%). The median best cosine is 0.99997. 49,869 features are alive, and there are 1.37 active features per token.
- **Training trajectory.** Codes start dense, with about 25,000 active features per token. They stall near 300 and then sparsify slowly towards one per token. Recovery only appears once codes are nearly one-hot. The run needed about four times its first 10-hour budget, so convergence time is part of the story.
- **Why it is a success.** 83% of tokens are coded by exactly one feature. The coefficient of that feature equals the embedding norm minus $\lambda/2$, the L1 shrinkage.
- **Where it falls short.** The unrecovered 7% lie in crowded regions of the embedding space. There a feature settles between two or three similar tokens, and other features make up the difference.

## Part 2: the failure modes

The tests on the 2,000-token subvocabulary ($m = 2{,}000$), showing that convergence to the embeddings is rare. Each test changes one choice and reports alive features, active features per token and recovery.

- **Top1.** Most features are dead at initialisation and never revive. Experiment 1 is the full-scale version: 0 of 39,914 dead features revived, and 16 tokens recovered.
- **Fixed-$k$ TopK.** A larger $k$ keeps features alive but gives up recovery. Recovery is gone by $k = 16$.
- **$k$-annealing.** Lowering $k$ from 32 to 1 does much better than Top1 but still loses most features at the final step from $k = 2$ to $k = 1$.
- **L1 penalty strength.** $\lambda = 0.03$ and $0.1$ leave codes too dense to recover tokens; $\lambda = 1$ kills half the features. Only a narrow range works.
- **Further tests**, to be added after the coming exploratory work.

The subset $\lambda = 0.3$ run is the reference point for Part 2. It is the subset counterpart of the success mode, so each failure can be read against it.

`docs/failure_modes.md` is the running source for this part.
