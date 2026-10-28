"""Experiment 10: a ReLU SAE with no sparsity penalty (lambda = 0), as a control
(plan: docs/experiment_10_relu_no_penalty.md)

One command runs everything:

    python experiments/exp10_relu_no_penalty/run.py

It trains a ReLU SAE with m = V on the 2,000 most frequent tokens with the
L1 coefficient set to zero, so the loss is the reconstruction error alone.
Everything else matches the best cell of Experiment 4 (lambda = 0.03): batch
size 256, the Gao learning rate, seed 0, 8 million (2^23) training tokens,
which is 32,768 steps. Decoder recovery and the spread of active features per
token are measured every half million tokens.

Without a penalty, every exact reconstruction has zero loss, so nothing in
the loss singles out the embeddings (Appendix A of docs/write_up_plan.md).
The question is whether training recovers them anyway.

Results go to runs/exp10/: the run's folder (eval_log.csv, result.json and
the final model.pt) and summary.csv. Ctrl-C saves and stops; running the same
command again resumes, losing at most 10 minutes. A heartbeat every minute
shows progress and the ETA. Everything is logged to runs/exp10/run.log.
"""

from sae_raw.evaluation import code_spread_fields, recovery_fields
from sae_raw.grid import make_cells, run_grid


def eval_extra(model, E):
    return {**recovery_fields(model, E), **code_spread_fields(model, E)}


def main(argv=None):
    """Run the experiment; returns None if it was interrupted."""
    return run_grid("experiment_10_relu_no_penalty", __doc__, "runs/exp10",
                    make_cells([0.0], [1.0], [256]), argv=argv, eval_extra=eval_extra)


if __name__ == "__main__":
    main()
