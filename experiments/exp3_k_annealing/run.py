"""Experiment 3: TopK with k annealed to 1 (plan: docs/experiment_3_k_annealing.md)

One command runs everything:

    caffeinate -i python experiments/exp3_k_annealing/run.py

It prepares the data if needed, times this machine at each k of the schedule
to fit --hours, and trains a TopK SAE with m = V on the GPT-2 embedding table
(uniform token sampling). k follows schedule B: it halves from 32 through 16,
8, 4 and 2 in equal stages, reaches 1 at 80% of training and stays at 1.
--k0, --end-frac and --shape change the schedule. The final model is a Top1
SAE, analysed for alive features, loss and decoder recovery.

A checkpoint is saved every 10 minutes. Ctrl-C saves and stops; running the
same command again resumes with the settings saved in the run directory. A
heartbeat line every minute shows progress, the current k and the ETA.
Everything is logged to the console and to <run-dir>/run.log.
"""

from sae_raw.run import run_experiment

SCHEDULE_B = {"k0": 32, "end_frac": 0.8, "shape": "halving"}
EXPERIMENT_1_RECOVERED = 16  # tokens recovered at cosine > 0.9 by the Top1 run of Experiment 1


def report(result, log):
    n = result["n_tokens"]
    log(f"Result: {result['n_alive']:,} of {n:,} features alive at the end (k = 1). "
        f"{result['n_recovered_cos0.9']:,} tokens recovered at cosine > 0.9 and "
        f"{result['n_recovered_cos0.99']:,} at cosine > 0.99, against "
        f"{EXPERIMENT_1_RECOVERED} at cosine > 0.9 in Experiment 1.")


if __name__ == "__main__":
    run_experiment("experiment_3_k_annealing", __doc__, "runs/exp3", model="topk",
                   report=report, schedule=SCHEDULE_B)
