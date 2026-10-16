# sae-raw-embeddings

Can a sparse autoencoder (SAE), trained by gradient descent from random initialisation, recover the raw token embeddings of a pretrained language model?

The setting is chosen so that an exact answer exists. We train on the rows of GPT-2 Small's input embedding table, with $V = 50{,}257$ rows in $d = 768$ dimensions and a dictionary of $m = V$ features. Decoder rows have unit norm and there is no decoder bias. For a Top1 SAE ($k = 1$), the zero-loss dictionary is then the set of normalised token embeddings, up to permutation. The question is whether training finds it.

## Experiments

Each experiment has a plan in `docs/` and its code in its own folder under `experiments/`.

- **Experiment 1** (plan `docs/experiment_1_top1.md`, code `experiments/exp1_top1/`, results `docs/experiment_1_results.md`): a Top1 SAE. Most features are dead at initialisation, and a dead feature gets no gradient. The experiment measures what percentage of the features dead at initialisation revive during training. Result: none revived, and only 259 features were alive at the end.
- **Experiment 2** (plan `docs/experiment_2_l1_relu.md`, code `experiments/exp2_l1_relu/`, not written yet): an L1-penalised ReLU SAE with the same data, dictionary and initialisation. It runs if fewer than 80% of the dead features revive in Experiment 1. Success is measured by decoder recovery, since the loss cannot reach zero.
- **Experiment 3** (plan `docs/experiment_3_k_annealing.md`, planning): as Experiment 1, but with a TopK encoder whose $k$ starts large, to keep features alive, and is lowered to 1 during training, to push towards one feature per token. A fixed $k > 1$ was considered first and dropped, because subset measurements showed it keeps features alive only by giving up recovery.

`docs/failure_modes.md` keeps a running table of the attempts that failed to recover the embeddings, with an interpretation of each.

Shared choices:

- Only the random-start SAE is run. The embedding-initialised control and the classical dictionary-learning baseline from the original plan are dropped for now.
- Token frequency plays no part in the design. Training tokens are sampled uniformly from the whole vocabulary, and no tokens are excluded from training or analysis.
- Frequency weighting is kept as a toggle, off by default. With `--frequency-weighted`, tokens are sampled in proportion to their English Wikipedia count (GPT-2 tokenisation), and the evaluation is weighted the same way. Zero-count tokens are then never sampled.

## Layout

```
sae_raw/              shared library code
  data.py             GPT-2 embedding table, Wikipedia token counts, diagnostics, subsets
  model.py            Top1SAE (Experiment 1), ReLUSAE (Experiment 2) and TopKSAE
                      (Experiment 3): shared initialisation and unit-norm decoder
  train.py            samplers and the resumable training loop (checkpoints, heartbeat)
  evaluation.py       full-vocabulary summary, revival of dead features, decoder recovery
  run.py              helpers for the experiment scripts: logging, timing, run settings
experiments/
  exp1_top1/run.py    runs all of Experiment 1
  exp2_l1_relu/       Experiment 2 (script to be written)
scripts/
  prepare_data.py     prepare the data on its own (the experiment scripts also do this)
docs/
  experiment_1_top1.md      plan: how many dead Top1 features revive?
  experiment_1_results.md   results and interpretation of Experiment 1
  failure_modes.md          running table of failed attempts and why they failed
  experiment_2_l1_relu.md   plan: L1-penalised ReLU SAE (runs if revival is below 80%)
  experiment_3_k_annealing.md  plan: TopK with k lowered to 1 during training
  planning/                 original planning documents, kept unchanged as a record
tests/                pytest suite
data/                 generated, not tracked by git
runs/                 generated, not tracked by git
```

## Setup

```
pip install -e ".[dev,train]"
pytest
```

The data is prepared by the experiment scripts if `data/` is missing, or on its own with `python scripts/prepare_data.py`. This downloads GPT-2 and reads English Wikipedia (snapshot `20231101.en`) directly from its parquet shards. Row groups of about 1,000 articles are visited in a seeded random order across all shards, and only the ones used are downloaded. By default it counts about 200M tokens (`--max-tokens` changes this). The counts are used only by the frequency toggle and by `--subset`.

Data preparation does not need torch. Training does. On an Intel Mac, PyTorch wheels stop at version 2.2.2 (Python 3.12 or earlier, NumPy below 2), so the training code avoids newer torch APIs.

## Running Experiment 1

```
caffeinate -i python experiments/exp1_top1/run.py
```

`caffeinate -i` keeps the Mac awake. The script does every step:

1. It prepares the data if `data/` does not exist yet.
2. It times a few training steps on this machine and chooses the number of steps that fits in `--hours` (default 10, including evaluations and the final analysis). `--steps` overrides this.
3. It trains, with a full-vocabulary evaluation at step 0 and then `--n-evals` more (default 20).
4. It analyses the result and states whether revival is above or below the 80% threshold.

`python experiments/exp1_top1/run.py --help` lists all options.

**Interrupting and resuming.** A checkpoint is saved every 10 minutes. On Ctrl-C the script finishes the current step, saves a checkpoint and stops. A second Ctrl-C aborts without saving. Running the same command again resumes from the checkpoint with the settings saved in the run directory, and gives the same result as an uninterrupted run. `--fresh` discards the earlier run and starts again.

**Logging.** Everything goes to the console and to `run.log` in the run directory. A heartbeat line every minute shows the step, the time per step, the elapsed time, the ETA for the whole run (with its clock time), the minibatch loss, and the last evaluation's relative loss, alive count and revival. Each evaluation is also logged in full.

**Run directory** (`--run-dir`, default `runs/exp1`, about 1.5 GB):

- `config.json`: the settings, including the measured step and evaluation times;
- `init.pt`: the parameters at initialisation, saved before the first step;
- `checkpoint.pt`: everything needed to resume (about three times the model size);
- `model.pt`: the parameters at the latest evaluation;
- `eval_log.csv`: one row per evaluation;
- `analysis.json`: the final report;
- `run.log`: the full log, appended to on every invocation.

Each evaluation reports:

- `mse` and `rel_loss`: the reconstruction error, and that error relative to reconstructing zero (1 means no better than zero);
- `mean_l0`: the mean number of active features per token (at most 1 for Top1);
- `n_alive`: features active on at least one token, and `n_zero_code`: tokens with no active feature;
- `n_dead_at_init`, `n_revived` and `revival_pct`: features that were dead at step 0, and how many of them have since received gradient.

A feature counts as revived if its parameters differ from their initial values by more than $10^{-5}$. A feature that is never active receives no gradient, so it stays at its initial values up to rounding in the decoder renormalisation (around $10^{-7}$). One Adam step moves a parameter by about the learning rate.

The final analysis recomputes revival from `init.pt` and `model.pt`. It also measures decoder recovery: for each token, the decoder row with the highest cosine to its normalised embedding. A token counts as recovered if that row is a mutual nearest neighbour and the cosine is above a threshold (0.9 and 0.99). A full one-to-one assignment over a $V \times m$ matrix is too large, so mutual nearest neighbours stand in for it.

`--subset N` trains on the N most frequent tokens with $m = N$, for quick debugging runs (use a separate `--run-dir`). The learning rate defaults to $2\times10^{-4}/\sqrt{m/2^{14}}$ (the scaling rule of Gao et al.), about $1.1\times10^{-4}$ at $m = 50{,}257$.

## Model choices

- **Shared:** random unit-norm Gaussian decoder rows, encoder copied from the decoder, encoder bias zero. No information from the embeddings is used, and both models start from identical parameters for the same seed. The gradient component along each decoder row is removed before each optimiser step, and rows are renormalised after it. Adam with default betas and epsilon, one run per seed, no sweep.
- **Top1** (`Top1SAE`, Experiment 1): the largest ReLU pre-activation wins, and its score is the coefficient. If no pre-activation is positive, the code and the reconstruction are zero. Ties go to the first index. Loss: MSE.
- **TopK** (`TopKSAE`, Experiment 3): the $k$ largest pre-activations are selected and passed through a ReLU, so a selected feature with a non-positive pre-activation is inactive. With $k = 1$ it matches Top1. Loss: MSE.
- **L1 ReLU** (`ReLUSAE`, Experiment 2): $f(x) = \mathrm{ReLU}(W_{\mathrm{enc}}x + b_{\mathrm{enc}})$, and the loss is MSE plus $\lambda \sum_j f_j$. The decoder is dense, so each step costs more than a Top1 step at the same batch size.
- The exact solution ($W_{\mathrm{dec}} = W_{\mathrm{enc}} = $ normalised embeddings) is only used in tests.

## Possible future directions

Ideas to revisit after the experiments above. Each one changes the training recipe or the setting, so each would be reported as a labelled ablation.

- **Encoder-side mean centring.** Initialise the encoder bias as $b_{\mathrm{enc},j} = -w_j^\top\mu$, where $\mu$ is the data mean, and keep the decoder bias at zero. This removes the shared mean component that makes a few features win most tokens. It leaves the target dictionary unchanged.

## Status

- [x] Data preparation
- [x] Top1, TopK and L1 ReLU models
- [x] Resumable training loop with revival tracking, checkpoints and heartbeat
- [x] Final analysis: revival and decoder recovery
- [x] Experiment 1 script (budget: about 10 hours on the laptop)
- [x] Experiment 1 full run (results in `docs/experiment_1_results.md`)
- [ ] Experiment 2 design questions ($\lambda$, learning rate, recovery threshold)
- [ ] Experiment 2 script
- [ ] Experiment 3: the $k$ schedule, then the script
