# Failure modes

Goal: train an SAE with $m = V$ features on the GPT-2 Small token embedding table, from random initialisation, so that its features recover the raw token embeddings.

This file records each attempt that fails to reach the goal, with what went wrong and why we think it did. Each row is one attempt: a full experiment, or a cheaper measurement that ruled a design out before a full run. The details are in the linked document.

| # | Setup | What happened | Interpretation | Write-up |
|---|---|---|---|---|
| 1 | Top1 SAE ($k = 1$), $m = V$, unit-norm decoder rows, zero decoder bias, MSE loss. Random initialisation, uniform token sampling, Adam, 16,820 steps. | 39,914 of 50,257 features were dead at initialisation and none revived. The alive features collapsed from 10,343 to 259 within the first 841 steps. Relative loss ended at 0.654, and only 16 tokens were recovered at cosine above 0.9. | Failure of the Top1 training regime. With $k = 1$ only the winning feature receives gradient, and the cutoff is too harsh for features to stay alive. | `docs/experiment_1_results.md` |
| 2 | TopK SAE with fixed $k \in \{2, 4, 8, 16, 32\}$, otherwise as row 1. Measured on the 2,000 most frequent tokens with $m = 2{,}000$, not run at full scale. | Raising $k$ kept more features alive (74% at $k = 16$, against 19% at $k = 1$) and lowered the loss. But tokens recovered at cosine above 0.9 fell from 222 at $k = 1$ to 0 at $k = 16$. Every run used all $k$ slots for every token. | A fixed $k$ cannot both keep features alive and keep codes one-hot. With $k > 1$ the loss rewards spreading each token over several features, so features become blends rather than embeddings. | `docs/experiment_3_k_annealing.md` (measurements) |
