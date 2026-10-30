# Can a sparse autoencoder find the token embeddings?

Code, results and paper for a toy model of sparse autoencoder (SAE) training with a known answer.

We train SAEs on the raw token embedding table of GPT-2 Small ($V = 50{,}257$ vectors in $d = 768$ dimensions), with exactly one feature per token ($m = V$). With no bottleneck, the natural target is the embedding table itself: one feature along each token's embedding. We train from random initialisation and measure how much of the target training recovers. A token counts as recovered if some decoder row is its mutual nearest neighbour at a cosine above 0.9.

**Paper:** [`paper/main.pdf`](paper/main.pdf). Main findings:

- **TopK SAEs** fail with any fixed $k$. Lowering $k$ during training to 1 succeeds, at a high training cost: on a 2,000-token subvocabulary, recovery rises by about 3 percentage points per doubling of training, to 89% at the longest run.
- **ReLU SAEs with an L1 penalty** recover 93% of the full table, and up to 98% of the subvocabulary, across a wide region of settings. The penalty must be weak, but not zero.
- **With training scaled in proportion to the vocabulary size, recovery does not depend on it**, for both architectures.

## Repository layout

```
paper/                the paper: main.tex, refs.bib, main.pdf, figures/
  make_figures.py     makes every figure in paper/figures/ from the results in runs/
sae_raw/              library code shared by the experiments
  data.py             GPT-2 embedding table, Wikipedia token counts, subvocabularies
  model.py            Top1, TopK and L1 ReLU SAEs with a unit-norm decoder
  train.py            resumable training loop, k schedules, samplers
  evaluation.py       full-vocabulary evaluation: loss, alive features, decoder recovery
  run.py              runner for single long runs (Experiments 1 to 3)
  grid.py             runner for grids of shorter runs (Experiments 4 and 7 to 10)
experiments/expN_*/   one folder per experiment, each with a single run.py
scripts/
  prepare_data.py     downloads the embedding table (run automatically when needed)
  code_spread.py      active features per token in the best ReLU SAEs (paper Table 7)
runs/                 results: one folder per experiment (CSV and JSON tables)
data/                 token counts used to choose the subvocabularies
docs/                 working notes: one document per experiment (see docs/README.md)
tests/                pytest suite
prompt_history/       the full conversation in which this project was developed
```

## Setup

```
pip install -e ".[train,paper,dev]"
pytest
```

The code runs on a CPU. It was developed with Python 3.12 and PyTorch 2.2.2 on an Intel Mac, and also runs with recent versions. All commands below are run from the repository root.

**Data.** `python scripts/prepare_data.py` downloads GPT-2 Small's input embedding table to `data/embeddings.npy` (148 MB, not tracked). Every experiment script does this itself if the file is missing. The subvocabularies are the $V$ most frequent tokens in English Wikipedia (snapshot `20231101.en`). The counts, from a seeded random sample of about 200 million tokens, are tracked in `data/token_counts.npy`, with their provenance in `data/meta.json`. `--recount` counts them again. Frequency only chooses the subvocabularies: training samples tokens uniformly, and evaluation weights every token equally.

## Reproducing the paper

**Figures.** The results of every run are tracked under `runs/`, so the figures can be rebuilt without training anything:

```
python paper/make_figures.py
cd paper && latexmk -pdf main.tex
```

A few numbers in the figure script come from runs whose logs are not tracked (the early 2k-subset pilots, and summary tables from the experiment documents); the script marks each one.

**Experiments.** Each experiment runs from a single command. Times are for the laptop used (Intel CPU), except those marked as measured in a cloud sandbox. Experiments 1 to 3 use batch size 4,096 (Experiment 2: 1,024) and seed 0 by default, so the commands above repeat the paper's runs.

| Experiment | Command | Time | Paper |
|---|---|---|---|
| 1. Top1, full table | `python experiments/exp1_top1/run.py --steps 16820` | 8 h | Section 3.2 |
| 2. L1 ReLU ($\lambda = 0.3$), full table | `python experiments/exp2_l1_relu/run.py --steps 85100` | 47 h | Sections 4, 5.5 |
| 3. TopK annealed to 1, full table | `python experiments/exp3_k_annealing/run.py --steps 15980` | 10 h | Sections 3, 5.5 |
| 4. L1 ReLU grid, 2k subset | `python experiments/exp4_relu_grid/run.py` | 10 h | Sections 4.3, 4.4 |
| 5. Alive at initialisation | `python experiments/exp5_alive_at_init/run.py` (and `--order random`, see its docstring) | 15 min (sandbox) | Sections 5.3, 5.4 |
| 6. Convex-hull vertices | `python experiments/exp6_hull_vertices/run.py` | 25 min (sandbox) | Section 4.2, Appendix B |
| 7. TopK with more training, 2k subset | `python experiments/exp7_topk_lr_steps/run.py` | 8 h | Section 3 |
| 8. TopK at 5k, 10k, 20k tokens | `python experiments/exp8_topk_sizes/run.py` | 5 h | Section 5.3 |
| 9. L1 ReLU at 5k, 10k, 20k tokens | `python experiments/exp9_relu_sizes/run.py` | 17 h | Section 5.2 |
| 10. No penalty, 2k subset | `python experiments/exp10_relu_no_penalty/run.py` | 20 min (sandbox) | Section 4.4 |

Experiment 6 also tabulates recovery against margin if Experiment 2's final model is in `runs/exp2/`. After Experiments 2 and 4, `python scripts/code_spread.py` gives the number of active features per token in the best ReLU SAEs, with and without a 10% cutoff (Section 4.3, Appendix E.2).

**Pilots on the 2k subset.** The early TopK pilots (fixed $k$ and the annealing schedules, Figure 2) used the scripts of Experiments 1 and 3 on the 2,000 most frequent tokens. For example:

```
# Top1 throughout, 4,000 steps
python experiments/exp1_top1/run.py --subset 2000 --steps 4000 --batch-size 512 --run-dir runs/pilots/top1
# fixed k = 16, 3,000 steps
python experiments/exp3_k_annealing/run.py --subset 2000 --steps 3000 --batch-size 512 --stages 16:1 --run-dir runs/pilots/k16
# halving from k0 = 32, reaching k = 1 at 80% of training
python experiments/exp3_k_annealing/run.py --subset 2000 --steps 4000 --batch-size 512 \
    --stages 32:0.16,16:0.16,8:0.16,4:0.16,2:0.16,1:0.2 --run-dir runs/pilots/halving32
```

The results of these pilots are recorded in `docs/experiment_3_k_annealing.md`, not as tables under `runs/`. The AuxK pilots (Section 3.4) used exploratory code that was not kept.

## How the scripts work

**Interrupting and resuming.** Every script saves a checkpoint every 10 minutes. On Ctrl-C it finishes the current step, saves and stops; a second Ctrl-C aborts. Running the same command again resumes and gives the same result as an uninterrupted run. A heartbeat line every minute shows progress and the ETA, and everything is logged to `run.log` in the run's folder.

**Single runs (Experiments 1 to 3).** Without `--steps`, the script times a few steps and fits the run to `--hours` (default 10). `--extend-steps N` continues a finished run; for a $k$ schedule, the part not yet reached is stretched over the new length. `--subset N` trains on the $N$ most frequent tokens with $m = N$. `--help` lists all options. Each run folder holds `config.json`, `eval_log.csv` (one row per evaluation) and `analysis.json` (the final report), and, while it runs, the model weights and checkpoints.

**Grids (Experiments 4 and 7 to 10).** One folder per cell, each with `eval_log.csv` and `result.json`, and a `summary.csv` with one row per cell and evaluation. Experiments 8 and 9 run one grid per vocabulary size (`--sizes`). Experiment 9's runs keep their checkpoints so they can be extended with `--extend CELL STEPS`.

**Evaluation fields.** `rel_loss` is the squared reconstruction error relative to the squared norm of the tokens. `mean_l0` is the mean number of active features per token. `n_alive` counts features active on at least one token. `n_recovered_cos0.9` and `n_recovered_cos0.99` count recovered tokens. For TopK runs, `n_top1` counts features that are the top feature of some token: the features that would survive a drop to $k = 1$.

## Model choices

- **Shared:** $m = V$ features, unit-norm decoder rows, no decoder bias. Decoder rows start as random Gaussian directions, the encoder starts as the transpose of the decoder, and the encoder bias starts at zero; no information from the embeddings is used. The gradient component along each decoder row is removed before each step, and rows are renormalised after it.
- **Training:** Adam at a constant learning rate $2\times10^{-4}/\sqrt{m/2^{14}}$ (the "Gao rate"). The $1/\sqrt{m}$ scaling follows Gao et al. (2024); the constant is our normalisation. Minibatches are drawn uniformly from the vocabulary, with replacement. One seed per run unless stated.
- **Top1 / TopK:** the $k$ largest pre-activations are kept and passed through a ReLU. In Experiment 3 and later TopK runs, $k$ follows a schedule down to 1.
- **L1 ReLU:** coefficients $f = \mathrm{ReLU}(W_{\mathrm{enc}}x + b_{\mathrm{enc}})$, loss $\lvert x - \hat x\rvert^2 + \lambda \sum_j f_j$.
- **Recovery:** mutual nearest neighbours between tokens and decoder rows by cosine, at thresholds 0.9 and 0.99. A full one-to-one assignment is too large to compute at $V = 50{,}257$.

## Prompt history

This project was developed in collaboration with Claude (Anthropic). `prompt_history/prompt_history.pdf` is the full conversation, kept for transparency.

## Licence

MIT; see `LICENSE`.
