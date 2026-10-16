# Failure modes

Goal: train an SAE with $m = V$ features on the GPT-2 Small token embedding table, from random initialisation, so that its features recover the raw token embeddings.

This file records each attempt that fails to reach the goal, with what went wrong and why we think it did. Each row is one experiment. The details are in the linked write-up.

| # | Setup | What happened | Interpretation | Write-up |
|---|---|---|---|---|
| 1 | Top1 SAE ($k = 1$), $m = V$, unit-norm decoder rows, zero decoder bias, MSE loss. Random initialisation, uniform token sampling, Adam, 16,820 steps. | 39,914 of 50,257 features were dead at initialisation and none revived. The alive features collapsed from 10,343 to 259 within the first 841 steps. Relative loss ended at 0.654, and only 16 tokens were recovered at cosine above 0.9. | Failure of the Top1 training regime. With $k = 1$ only the winning feature receives gradient, and the cutoff is too harsh for features to stay alive. | `docs/experiment_1_results.md` |
