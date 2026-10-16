# sae-raw-embeddings

Can a sparse autoencoder (SAE), trained by gradient descent from random initialisation, recover the raw token embeddings of a pretrained language model?

The setting is chosen so that an exact answer exists. We train on the rows of GPT-2 Small's input embedding table, with $V = 50{,}257$ rows in $d = 768$ dimensions and a dictionary of $m = V$ features. Decoder rows have unit norm and there is no decoder bias. For a Top1 SAE ($k = 1$), the zero-loss dictionary is then the set of normalised token embeddings, up to permutation. The question is whether training finds it.

## Experiments

Each experiment has a plan in `docs/` and its code in its own folder under `experiments/`.

- **Experiment 1** (plan `docs/experiment_1_top1.md`, results `docs/experiment_1_results.md`, code `experiments/exp1_top1/`): a Top1 SAE. Most features are dead at initialisation, and a dead feature gets no gradient. The experiment measured what percentage of the features dead at initialisation revive during training. Result: none revived, and only 259 features were alive at the end.
- **Experiment 2** (plan `docs/experiment_2_l1_relu.md`, code `experiments/exp2_l1_relu/`, script not written yet): an L1-penalised ReLU SAE with the same data, dictionary and initialisation. It goes ahead because revival in Experiment 1 was below 80%. Success is measured by decoder recovery, since the loss cannot reach zero.
- **Experiment 3** (plan `docs/experiment_3_k_annealing.md`, code `experiments/exp3_k_annealing/`, ready to run): as Experiment 1, but with a TopK encoder whose $k$ starts at 32, to keep features alive, and is lowered to 1 during training, to push towards one feature per token. A fixed $k > 1$ was considered first and dropped, because subset measurements showed it keeps features alive only by giving up recovery. On the subset, the chosen schedule recovered six times as many tokens as Top1.

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
  train.py            samplers, k schedules, and the resumable training loop
                      (checkpoints, heartbeat)
  evaluation.py       full-vocabulary summary, revival of dead features, decoder recovery
  run.py              the runner behind the experiment scripts: data, timing, training,
                      analysis, logging
experiments/
  exp1_top1/run.py         runs all of Experiment 1
  exp2_l1_relu/            Experiment 2 (script to be written)
  exp3_k_annealing/run.py  runs all of Experiment 3
scripts/
  prepare_data.py     prepare the data on its own (the experiment scripts also do this)
docs/
  experiment_1_top1.md         plan: how many dead Top1 features revive?
  experiment_1_results.md      results and interpretation of Experiment 1
  experiment_2_l1_relu.md      plan: L1-penalised ReLU SAE
  experiment_3_k_annealing.md  plan: TopK with k lowered to 1 during training
  failure_modes.md             running table of failed attempts and why they failed
  planning/                    original planning documents, kept unchanged as a record
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

## Running an experiment

Each experiment runs from a single script:

```
caffeinate -i python experiments/exp1_top1/run.py
caffeinate -i python experiments/exp3_k_annealing/run.py
```

`caffeinate -i` keeps the Mac awake. Each script does every step:

1. It prepares the data if `data/` does not exist yet.
2. It times a few training steps on this machine and chooses the number of steps that fits in `--hours` (default 10, including evaluations and the final analysis). `--steps` overrides this. For Experiment 3 it times each value of $k$ in the schedule, since steps at large $k$ are slower.
3. It trains, with a full-vocabulary evaluation at step 0 and then `--n-evals` more (default 20).
4. It analyses the result and logs the experiment's conclusion.

`--help` lists all options. Experiment 3 also takes `--k0`, `--end-frac` and `--shape` to change the schedule.

**Interrupting and resuming.** A checkpoint is saved every 10 minutes. On Ctrl-C the script finishes the current step, saves a checkpoint and stops. A second Ctrl-C aborts without saving. Running the same command again resumes from the checkpoint with the settings saved in the run directory, and gives the same result as an uninterrupted run. `--fresh` discards the earlier run and starts again.

**Logging.** Everything goes to the console and to `run.log` in the run directory. A heartbeat line every minute shows the step, the current $k$ (Experiment 3), the time per step, the elapsed time, the ETA for the whole run (with its clock time), the minibatch loss, and the last evaluation's relative loss, alive count and revival. Each evaluation is also logged in full.

**Run directory** (`--run-dir`, default `runs/exp1` or `runs/exp3`, about 1.5 GB):

- `config.json`: the settings, including the measured step and evaluation times;
- `init.pt`: the parameters at initialisation, saved before the first step;
- `checkpoint.pt`: everything needed to resume (about three times the model size);
- `model.pt`: the parameters at the latest evaluation;
- `eval_log.csv`: one row per evaluation;
- `analysis.json`: the final report;
- `run.log`: the full log, appended to on every invocation.

Each evaluation reports:

- `k`: the number of features selected per token at that point;
- `mse` and `rel_loss`: the reconstruction error, and that error relative to reconstructing zero (1 means no better than zero);
- `mean_l0`: the mean number of active features per token (at most $k$);
- `n_alive`: features active on at least one token, and `n_zero_code`: tokens with no active feature;
- `n_dead_at_init`, `n_revived` and `revival_pct`: features that were dead at step 0, and how many of them have since received gradient.

A feature counts as revived if its parameters differ from their initial values by more than $10^{-5}$. A feature that is never active receives no gradient, so it stays at its initial values up to rounding in the decoder renormalisation (around $10^{-7}$). One Adam step moves a parameter by about the learning rate.

The final analysis recomputes revival from `init.pt` and `model.pt`. It also measures decoder recovery: for each token, the decoder row with the highest cosine to its normalised embedding. A token counts as recovered if that row is a mutual nearest neighbour and the cosine is above a threshold (0.9 and 0.99). A full one-to-one assignment over a $V \times m$ matrix is too large, so mutual nearest neighbours stand in for it.

`--subset N` trains on the N most frequent tokens with $m = N$, for quick debugging runs (use a separate `--run-dir`). The learning rate defaults to $2\times10^{-4}/\sqrt{m/2^{14}}$ (the scaling rule of Gao et al.), about $1.1\times10^{-4}$ at $m = 50{,}257$.

## Model choices

- **Shared:** random unit-norm Gaussian decoder rows, encoder copied from the decoder, encoder bias zero. No information from the embeddings is used, and all models start from identical parameters for the same seed. The gradient component along each decoder row is removed before each optimiser step, and rows are renormalised after it. Adam with default betas and epsilon, one run per seed, no sweep.
- **Top1** (`Top1SAE`, Experiment 1): the largest ReLU pre-activation wins, and its score is the coefficient. If no pre-activation is positive, the code and the reconstruction are zero. Ties go to the first index. Loss: MSE.
- **TopK** (`TopKSAE`, Experiment 3): the $k$ largest pre-activations are selected and passed through a ReLU, so a selected feature with a non-positive pre-activation is inactive. With $k = 1$ it matches Top1. In Experiment 3, $k$ follows a schedule (`k_schedule` in `sae_raw/train.py`). Loss: MSE.
- **L1 ReLU** (`ReLUSAE`, Experiment 2): $f(x) = \mathrm{ReLU}(W_{\mathrm{enc}}x + b_{\mathrm{enc}})$, and the loss is MSE plus $\lambda \sum_j f_j$. The decoder is dense, so each step costs more than a Top1 step at the same batch size.
- The exact solution ($W_{\mathrm{dec}} = W_{\mathrm{enc}} = $ normalised embeddings) is only used in tests.

## Possible future directions

Ideas to revisit after the experiments above. Each one changes the training recipe or the setting, so each would be reported as a labelled ablation.

- **Longer time at $k = 2$ in Experiment 3's schedule.** On the subset, most features that die are lost at the final step from $k = 2$ to $k = 1$. A longer stage at $k = 2$ might keep more of them.
- **Encoder-side mean centring.** Initialise the encoder bias as $b_{\mathrm{enc},j} = -w_j^\top\mu$, where $\mu$ is the data mean, and keep the decoder bias at zero. This removes the shared mean component that makes a few features win most tokens. It leaves the target dictionary unchanged.

## Status

- [x] Data preparation
- [x] Top1, TopK and L1 ReLU models
- [x] Resumable training loop with revival tracking, checkpoints, heartbeat and $k$ schedules
- [x] Final analysis: revival and decoder recovery
- [x] Experiment 1: script and full run (results in `docs/experiment_1_results.md`)
- [x] Experiment 3: schedule chosen (halving from 32, reaching 1 at 80% of training) and script written
- [ ] Experiment 3 full run
- [ ] Experiment 2: design questions ($\lambda$, learning rate, recovery threshold), then the script
