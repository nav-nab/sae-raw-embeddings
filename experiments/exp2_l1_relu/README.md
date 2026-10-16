# Experiment 2: L1-penalised ReLU SAE

Plan: `docs/experiment_2_l1_relu.md`.

The script for this experiment, `run.py`, is not written yet. It will be written once the open design questions are settled: $\lambda$, the learning rate and the recovery threshold. Like `experiments/exp1_top1/run.py` and `experiments/exp3_k_annealing/run.py`, it will describe the experiment and call the shared runner in `sae_raw/run.py`. It will use the `ReLUSAE` model from the shared library (`sae_raw/model.py`), which is already written and tested.
