"""Experiment 2: an L1-penalised ReLU SAE (plan: docs/experiment_2_l1_relu.md)

One command runs everything:

    caffeinate -i python experiments/exp2_l1_relu/run.py

It prepares the data if needed, times this machine to fit --hours, and trains
a ReLU SAE with m = V on the GPT-2 embedding table (uniform token sampling).
The loss is MSE plus an L1 penalty on the activations, with coefficient
--l1-coef (0.3, chosen by a subset sweep). The batch size is 1024, since the
dense decoder makes large batches slow. The result is analysed for alive
features, active features per token, loss and decoder recovery.

A checkpoint is saved every 10 minutes. Ctrl-C saves and stops; running the
same command again resumes with the settings saved in the run directory. A
heartbeat line every minute shows progress and the ETA. Everything is logged
to the console and to <run-dir>/run.log.
"""

from sae_raw.run import run_experiment

L1_COEF = 0.3
EXPERIMENT_1_RECOVERED = 16  # tokens recovered at cosine > 0.9 by the Top1 run of Experiment 1


def report(result, log):
    log(f"Result: {result['n_alive']:,} of {result['n_tokens']:,} features alive, "
        f"{result['mean_l0']:.2f} active features per token on average. "
        f"{result['n_recovered_cos0.9']:,} tokens recovered at cosine > 0.9 and "
        f"{result['n_recovered_cos0.99']:,} at cosine > 0.99, against "
        f"{EXPERIMENT_1_RECOVERED} at cosine > 0.9 in Experiment 1.")


if __name__ == "__main__":
    run_experiment("experiment_2_l1_relu", __doc__, "runs/exp2", model="l1", report=report,
                   l1_coef=L1_COEF, batch_size=1024)
