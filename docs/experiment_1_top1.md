# Experiment 1: how many dead Top1 features revive?

Results: `docs/experiment_1_results.md`.

## Motivation

The project asks whether gradient descent can find a known global minimum. We train a Top1 SAE ($k = 1$) with $m = V$ features on the GPT-2 Small token embedding table. With unit-norm decoder rows and zero decoder bias, the zero-loss dictionary is the set of normalised embeddings. So the answer is known, and the question is whether training finds it.

Measurements at initialisation suggest a serious obstacle. Only about 10,000 of the 50,257 features win any token (10,343 for seed 0). The rest are dead. A Top1 SAE is a form of gain-shape k-means, and like k-means it gives no gradient to a feature that is never selected. A dead feature's parameters therefore stay exactly at their initial values. The cause is partly geometric. GPT-2 embeddings share a large mean component (mean pairwise cosine 0.27), so a few random directions win most tokens.

Dead features could still revive during training. A dead feature's score on a token is fixed, but the winner's score is not. If the winner moves away from a token, a dead feature may overtake it and start receiving gradient. Small-subset runs showed the number of used features falling during training. That is only a net count, though, and it does not rule out revival. This experiment measures revival directly at full scale.

## Question

What percentage of the features that are dead at initialisation become alive during training? For the trained dictionary to match the raw embeddings, this percentage has to be very high. We expect it to be low.

## Design

- **Data:** the GPT-2 Small input embedding table, with $V = 50{,}257$ rows and $d = 768$. Tokens are sampled uniformly. Frequency weighting is off.
- **Model:** a Top1 SAE with $m = V$ features. The encoder picks the largest ReLU pre-activation. Its score, unthresholded, is the coefficient. If no pre-activation is positive, the code is zero. Decoder rows have unit norm, and the decoder bias is zero.
- **Loss:** plain MSE.
- **Initialisation:** random unit-norm Gaussian decoder rows. The encoder is a copy of the decoder, and the encoder bias is zero. No information from the embeddings is used. The initial parameters are saved to `init.pt`.
- **Training:** Adam with default betas and epsilon. The learning rate is $2\times10^{-4}/\sqrt{m/2^{14}} \approx 1.1\times10^{-4}$. Unit norm is kept by projecting out the gradient component along each decoder row and renormalising after each step. One seed. The run has a budget of about 10 hours on the laptop, and the step count is set from measured step times.
- **Logged at each evaluation:** loss, relative loss, the number of features that win at least one token, the number of tokens with a zero code, and the revival count so far.

## Measurement

- A feature is **dead at initialisation** if it wins no token under the initial parameters.
- A feature has **revived** if its parameters in `model.pt` differ from `init.pt` by more than $10^{-5}$. A feature that never won received no gradient, so it stays at its initial values up to rounding.
- The headline number is the **revival percentage**: the share of features dead at initialisation that have revived by the end. We also report the final number of alive features.

## How to run

```
caffeinate -i python experiments/exp1_top1/run.py
```

The script prepares the data if needed and times the machine to fit the 10-hour budget. It then trains with a checkpoint every 10 minutes and a heartbeat every minute, and finishes with the analysis. Ctrl-C saves and stops, and the same command resumes. Results go to `runs/exp1/`: `eval_log.csv` tracks the revival percentage at every evaluation, and `analysis.json` holds the final report. A feature counts as revived if its parameters moved by more than $10^{-5}$, well above the rounding error of decoder renormalisation and well below one Adam step.

## Next step

If the revival percentage is below 80%, we move on to Experiment 2. It replaces the Top1 encoder with an L1-penalised ReLU encoder. (The result was 0%. Experiment 3, a TopK SAE with $k$ annealed to 1, was added afterwards.)
