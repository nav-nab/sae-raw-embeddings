"""Experiment 3: TopK with k annealed to 1 (plan: docs/experiment_3_k_annealing.md)

One command runs everything:

    caffeinate -i python experiments/exp3_k_annealing/run.py

It prepares the data if needed, times this machine at each k of the schedule
to fit --hours, and trains a TopK SAE with m = V on the GPT-2 embedding table
(uniform token sampling). k follows the schedule with a long k = 2 stage:
128, 64, 32, 16, 8 and 4 share the first 40% of training equally, then k = 2
for 40% and k = 1 for the last 20%. --stages changes the schedule. The final
model is a Top1 SAE, analysed for alive features, loss and decoder recovery.

With --subset 2000 and --steps, --batch-size and --stages, the same script runs
the TopK pilots on the 2,000 most frequent tokens (see docs/reproducing.md).

A checkpoint is saved every 10 minutes. Ctrl-C saves and stops; running the
same command again resumes with the settings saved in the run directory. A
heartbeat line every minute shows progress, the current k and the ETA.
--extend-steps or --extend-hours lengthens the run: the part of the schedule
not yet reached is stretched over the new length. Everything is logged to the
console and to <run-dir>/run.log.
"""

from sae_raw.run import run_experiment

SCHEDULE = {"shape": "stages",
              "stages": [[k, 0.4 / 6] for k in (128, 64, 32, 16, 8, 4)] + [[2, 0.4], [1, 0.2]]}
EXPERIMENT_1_RECOVERED = 16  # tokens recovered at cosine > 0.9 by the Top1 run of Experiment 1


def report(result, log):
    n = result["n_tokens"]
    log(f"Result: {result['n_alive']:,} of {n:,} features alive at the end (k = 1). "
        f"{result['n_recovered_cos0.9']:,} tokens recovered at cosine > 0.9 and "
        f"{result['n_recovered_cos0.99']:,} at cosine > 0.99, against "
        f"{EXPERIMENT_1_RECOVERED} at cosine > 0.9 in Experiment 1.")


if __name__ == "__main__":
    run_experiment("experiment_3_k_annealing", __doc__, "runs/exp3", model="topk",
                   report=report, schedule=SCHEDULE)
