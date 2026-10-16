"""Experiment 1: how many dead Top1 features revive? (plan: docs/experiment_1_top1.md)

One command runs everything:

    caffeinate -i python experiments/exp1_top1/run.py

It prepares the data if needed, times this machine to fit --hours, trains a
Top1 SAE with m = V on the GPT-2 embedding table (uniform token sampling),
and analyses the result: the percentage of features dead at initialisation
that revived, the final loss and alive count, and decoder recovery.

A checkpoint is saved every 10 minutes. Ctrl-C saves and stops; running the
same command again resumes with the settings saved in the run directory. A
heartbeat line every minute shows progress and the ETA. Everything is logged
to the console and to <run-dir>/run.log.
"""

from sae_raw.run import run_experiment

REVIVAL_THRESHOLD = 80.0  # percent; below this we move on to Experiment 2


def report(result, log):
    pct = result["revival_pct"]
    below = pct < REVIVAL_THRESHOLD
    log(f"Result: {result['n_revived']:,} of {result['n_dead_at_init']:,} features dead at "
        f"initialisation revived ({pct:.2f}%), {'below' if below else 'at or above'} the "
        f"{REVIVAL_THRESHOLD:g}% threshold. "
        + ("Next: Experiment 2." if below else "Revival is high enough to stay with Top1."))


if __name__ == "__main__":
    run_experiment("experiment_1_top1", __doc__, "runs/exp1", model="top1", report=report)
