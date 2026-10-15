"""Experiment 1: how many dead Top1 features revive? (plan: docs/experiment_1_top1.md)

One command runs everything:

    caffeinate -i python experiments/exp1_top1/run.py

1. Prepares the data in --data-dir if it is not there yet.
2. Times a few training steps on this machine and chooses the number of steps
   that fits in --hours (unless --steps is given).
3. Trains a Top1 SAE with m = V on the GPT-2 embedding table, with uniform
   token sampling (--frequency-weighted switches to Wikipedia frequencies).
   A checkpoint is saved every 10 minutes. On Ctrl-C the script saves and
   stops; running the same command again resumes, using the settings saved
   in the run directory. A heartbeat line every minute shows progress and
   the ETA.
4. Analyses the result: the percentage of features dead at initialisation
   that revived, the final loss and alive count, and decoder recovery.

Everything is logged to the console and to <run-dir>/run.log. The run
directory also holds config.json, init.pt, checkpoint.pt, model.pt,
eval_log.csv and analysis.json.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from sae_raw import data, run
from sae_raw.evaluation import analyse
from sae_raw.train import INIT_FILE, MODEL_FILE, default_lr, fmt_duration, train

EXPERIMENT = "experiment_1_top1"
REVIVAL_THRESHOLD = 80.0  # percent; below this we move on to Experiment 2
ANALYSIS_EVALS = 3        # the final analysis costs about this many evaluations


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/exp1")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--hours", type=float, default=10.0, help="time budget for the whole run")
    p.add_argument("--steps", type=int, default=None,
                   help="number of training steps (default: chosen from --hours)")
    p.add_argument("--n-evals", type=int, default=20,
                   help="full-vocabulary evaluations during training, after step 0")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 2e-4 / sqrt(m / 2^14), the Gao et al. rule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subset", type=int, default=0,
                   help="debugging: train on the N most frequent tokens only, with m = N")
    p.add_argument("--frequency-weighted", action="store_true",
                   help="sample tokens by Wikipedia frequency and weight evaluation the same "
                        "way (off in the experiment design)")
    p.add_argument("--checkpoint-minutes", type=float, default=10.0)
    p.add_argument("--heartbeat-seconds", type=float, default=60.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fresh", action="store_true",
                   help="discard any earlier run in --run-dir and start again")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = run.setup_logging(run_dir)
    log("=" * 70)
    log(f"{EXPERIMENT}: run directory {run_dir}")

    if args.fresh:
        run.clear_run(run_dir)
        log("--fresh: removed the files of the earlier run.")

    resuming = (run_dir / run.CONFIG_FILE).exists()
    cfg = run.load_config(run_dir) if resuming else None
    data_dir = cfg["data_dir"] if resuming else args.data_dir
    if not data.is_prepared(data_dir):
        log(f"No prepared data in {data_dir}; preparing it now.")
        data.prepare(data_dir, log=log)

    if resuming:
        log("Found an earlier run: resuming with its saved settings "
            "(command-line training options are ignored; use --fresh to start again).")
        E, counts = data.load_prepared(data_dir, cfg["subset"])
    else:
        E, counts = data.load_prepared(data_dir, args.subset)
        V = E.shape[0]
        cfg = {
            "experiment": EXPERIMENT, "model": "top1", "l1_coef": None,
            "data_dir": args.data_dir, "subset": args.subset, "V": V, "d": E.shape[1],
            "batch_size": args.batch_size, "lr": args.lr or default_lr(V), "seed": args.seed,
            "frequency_weighted": args.frequency_weighted, "hours": args.hours, "n_evals": args.n_evals,
            "checkpoint_minutes": args.checkpoint_minutes,
            "heartbeat_seconds": args.heartbeat_seconds, "device": args.device,
            "python": platform.python_version(), "torch": torch.__version__,
            "threads": torch.get_num_threads(), "started": time.strftime("%Y-%m-%d %H:%M"),
        }
        log(f"Timing {args.batch_size}-token steps on {V:,} tokens "
            f"({torch.get_num_threads()} threads) ...")
        step_s, eval_s = run.time_training(E, args.batch_size, device=args.device)
        cfg.update(step_seconds=round(step_s, 3), eval_seconds=round(eval_s, 1))
        if args.steps is None:
            cfg["steps"], cfg["eval_every"] = run.plan_steps(
                args.hours, step_s, eval_s, args.n_evals, extra_evals=ANALYSIS_EVALS)
        else:
            cfg["steps"] = args.steps
            cfg["eval_every"] = max(args.steps // args.n_evals, 1)
        expected = cfg["steps"] * step_s + (args.n_evals + 1 + ANALYSIS_EVALS) * eval_s
        log(f"  {step_s:.2f} s per step, {eval_s:.0f} s per evaluation: "
            f"{cfg['steps']:,} steps, evaluation every {cfg['eval_every']:,}, "
            f"expected {fmt_duration(expected)}.")
        run.save_config(run_dir, cfg)

    log("Settings: " + json.dumps({k: cfg[k] for k in (
        "V", "batch_size", "lr", "seed", "steps", "eval_every", "subset",
        "frequency_weighted")}))
    train_counts = counts if cfg["frequency_weighted"] else None

    _, history, finished = train(
        E, steps=cfg["steps"], batch_size=cfg["batch_size"], lr=cfg["lr"], seed=cfg["seed"],
        eval_every=cfg["eval_every"], model="top1", counts=train_counts, device=cfg["device"],
        run_dir=run_dir,
        checkpoint_every=60 * cfg["checkpoint_minutes"],
        heartbeat_every=cfg["heartbeat_seconds"],
        extra_eta_seconds=ANALYSIS_EVALS * cfg["eval_seconds"], log=log)
    if not finished:
        return

    log("Training finished. Analysing ...")
    Et = torch.as_tensor(E, dtype=torch.float32)
    V, d = Et.shape
    init = run.load_model_file(run_dir / INIT_FILE, cfg, d, V)
    final = run.load_model_file(run_dir / MODEL_FILE, cfg, d, V)
    result = analyse(init, final, Et, run.token_weights(V, train_counts))
    result["training_hours"] = round(history[-1]["seconds"] / 3600, 2)
    (run_dir / run.ANALYSIS_FILE).write_text(json.dumps(result, indent=2))
    for k, v in result.items():
        log(f"  {k:>22}: {v:.4g}" if isinstance(v, float) else f"  {k:>22}: {v}")

    pct = result["revival_pct"]
    verdict = ("below" if pct < REVIVAL_THRESHOLD else "at or above")
    log(f"Result: {result['n_revived']:,} of {result['n_dead_at_init']:,} features dead at "
        f"initialisation revived ({pct:.2f}%), {verdict} the {REVIVAL_THRESHOLD:g}% threshold. "
        + ("Next: Experiment 2." if pct < REVIVAL_THRESHOLD else
           "Revival is high enough to stay with Top1."))


if __name__ == "__main__":
    main()
