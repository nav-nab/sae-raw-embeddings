"""The shared runner behind the experiment scripts (experiments/<experiment>/run.py).

Each experiment script describes one experiment (its model, schedule and how
to report the result) and calls run_experiment(). The runner does every step:

1. prepares the data if it is missing;
2. times a few training steps on this machine and chooses the number of steps
   that fits in --hours (unless --steps is given);
3. trains with checkpoints every 10 minutes and a heartbeat every minute.
   Ctrl-C saves and stops, and running the same command again resumes with
   the settings saved in the run directory;
4. analyses the result and writes analysis.json.

Everything is logged to the console and to <run-dir>/run.log.
"""

import argparse
import json
import logging
import platform
import shutil
import sys
import time
from pathlib import Path

import torch

from sae_raw import data
from sae_raw.evaluation import analyse
from sae_raw.model import build_model
from sae_raw.train import (CHECKPOINT_FILE, EVAL_LOG_FILE, INIT_FILE, MODEL_FILE, SNAPSHOT_DIR,
                           FrequencySampler, Trainer, UniformSampler, default_lr, fmt_duration,
                           k_schedule, k_stages, schedule_start_k, stretch_stages, train)

CONFIG_FILE = "config.json"
ANALYSIS_FILE = "analysis.json"
LOG_FILE = "run.log"
RUN_FILES = (CONFIG_FILE, INIT_FILE, CHECKPOINT_FILE, MODEL_FILE, EVAL_LOG_FILE, ANALYSIS_FILE)
ANALYSIS_EVALS = 3  # the final analysis costs about this many evaluations


def setup_logging(run_dir):
    """Log to the console and append to run_dir/run.log. Returns a log(message) function."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sae_raw.run")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(run_dir / LOG_FILE, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger.info


def clear_run(run_dir):
    """Remove the files of an earlier run in run_dir (the log is kept and appended to)."""
    for name in RUN_FILES:
        path = Path(run_dir) / name
        if path.exists():
            path.unlink()
    shutil.rmtree(Path(run_dir) / SNAPSHOT_DIR, ignore_errors=True)


def time_training(E, batch_size, model="top1", l1_coef=None, k=None, n_steps=3, device="cpu",
                  evaluate=True):
    """Seconds per training step and per full evaluation, on a throwaway model.

    With evaluate=False the evaluation is skipped and its time returned as None.
    """
    tr = Trainer(E, lr=default_lr(len(E)), seed=0, model=model, l1_coef=l1_coef, k=k,
                 device=device)
    tr.train_step(batch_size)  # warm-up
    t0 = time.time()
    for _ in range(n_steps):
        tr.train_step(batch_size)
    step_s = (time.time() - t0) / n_steps
    if not evaluate:
        return step_s, None
    t0 = time.time()
    tr.evaluate()
    return step_s, time.time() - t0


def plan_steps(hours, step_s, eval_s, n_evals, extra_evals=0):
    """Choose (steps, eval_every) so that training plus evaluations fit in the budget.

    steps is a multiple of eval_every, giving exactly n_evals evaluations
    after step 0. step_s is the mean time per step. extra_evals reserves time
    for work after training, counted in units of one evaluation.
    """
    budget = hours * 3600 - (n_evals + 1 + extra_evals) * eval_s
    eval_every = max(int(budget / step_s) // n_evals, 1)
    return eval_every * n_evals, eval_every


def pick_snapshot_steps(start, steps, eval_every, n):
    """n evaluation steps in (start, steps], evenly spread and including the last.

    Evaluations fall on multiples of eval_every and at the last step. If
    there are n or fewer of them, all are returned.
    """
    evals = [s for s in range(start + 1, steps + 1) if s % eval_every == 0 or s == steps]
    if n >= len(evals):
        return evals
    return [evals[round((j + 1) * len(evals) / n) - 1] for j in range(n)]


def mean_step_seconds(step_seconds_by_k, schedule, n=1000):
    """Mean step time over a schedule, given the step time at each k."""
    return sum(step_seconds_by_k[k_schedule(s, n, **schedule)] for s in range(1, n + 1)) / n


def schedule_ks(schedule, n=1000):
    """The distinct values of k a schedule passes through, largest first."""
    return sorted({k_schedule(s, n, **schedule) for s in range(1, n + 1)}, reverse=True)


def parse_stages(text):
    """Parse "64:0.08,2:0.4,1:0.52" into [[64, 0.08], [2, 0.4], [1, 0.52]]."""
    stages = []
    for part in text.split(","):
        k, frac = part.split(":")
        stages.append([int(k), float(frac)])
    if abs(sum(f for _, f in stages) - 1) > 1e-6:
        raise argparse.ArgumentTypeError("the stage fractions must sum to 1")
    return stages


def format_stages(stages):
    return ",".join(f"{k}:{frac:g}" for k, frac in stages)


def describe_stages(stages):
    """Readable layout of a schedule over a run, e.g. "k=64 to step 960, k=32 to ..."."""
    return ", ".join(f"k={k} to step {last:,}" for k, last in stages)


def load_config(run_dir):
    return json.loads((Path(run_dir) / CONFIG_FILE).read_text())


def save_config(run_dir, cfg):
    (Path(run_dir) / CONFIG_FILE).write_text(json.dumps(cfg, indent=2))


def token_weights(V, counts=None):
    """Evaluation weights: uniform over V tokens, or proportional to counts if given."""
    return UniformSampler(V, 0).p if counts is None else FrequencySampler(counts, 0).p


def load_model_file(path, cfg, d, m, k=None):
    """Load saved parameters into a model built from cfg (k overrides cfg["k"])."""
    model = build_model(cfg["model"], d, m, l1_coef=cfg.get("l1_coef"),
                        k=k if k is not None else cfg.get("k"))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def _parser(description, default_run_dir, schedule, l1_coef, batch_size):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=default_run_dir)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--hours", type=float, default=10.0, help="time budget for the whole run")
    p.add_argument("--steps", type=int, default=None,
                   help="number of training steps (default: chosen from --hours)")
    p.add_argument("--n-evals", type=int, default=20,
                   help="full-vocabulary evaluations during training, after step 0")
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 2e-4 / sqrt(m / 2^14), the Gao et al. rule")
    p.add_argument("--seed", type=int, default=0)
    if l1_coef is not None:
        p.add_argument("--l1-coef", type=float, default=l1_coef, help="L1 penalty coefficient")
    if schedule is not None:
        p.add_argument("--stages", type=parse_stages, default=schedule["stages"],
                       help="k schedule as k:fraction pairs in order, fractions summing to 1 "
                            f"(default {format_stages(schedule['stages'])})")
    p.add_argument("--subset", type=int, default=0,
                   help="debugging: train on the N most frequent tokens only, with m = N")
    p.add_argument("--frequency-weighted", action="store_true",
                   help="sample tokens by Wikipedia frequency and weight evaluation the same "
                        "way (off in the experiment design)")
    p.add_argument("--n-snapshots", type=int, default=10,
                   help="evaluations at which the parameters are kept in snapshots/, spread "
                        "evenly over the run (or over the new steps of an extension)")
    p.add_argument("--checkpoint-minutes", type=float, default=10.0)
    p.add_argument("--heartbeat-seconds", type=float, default=60.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--extend-steps", type=int, default=None,
                   help="continue an existing run for this many more steps (rounded up to a "
                        "whole number of evaluation intervals)")
    p.add_argument("--extend-hours", type=float, default=None,
                   help="continue an existing run for about this many more hours")
    p.add_argument("--fresh", action="store_true",
                   help="discard any earlier run in --run-dir and start again")
    return p


def run_experiment(name, description, default_run_dir, model, report, schedule=None,
                   l1_coef=None, batch_size=4096, argv=None):
    """Run one experiment end to end (see the module docstring).

    model is "top1", "topk" or "l1". For "topk", schedule is the default k
    schedule, {"shape": "stages", "stages": [[k, fraction], ...]}; for "l1",
    l1_coef is the default L1 coefficient. Both can be changed on the command
    line, as can batch_size.
    report(result, log) logs the experiment-specific conclusion.
    """
    args = _parser(description, default_run_dir, schedule, l1_coef, batch_size).parse_args(argv)
    run_dir = Path(args.run_dir)
    log = setup_logging(run_dir)
    log("=" * 70)
    log(f"{name}: run directory {run_dir}")

    if args.fresh:
        clear_run(run_dir)
        log("--fresh: removed the files of the earlier run.")

    resuming = (run_dir / CONFIG_FILE).exists()
    cfg = load_config(run_dir) if resuming else None
    data_dir = cfg["data_dir"] if resuming else args.data_dir
    if not data.is_prepared(data_dir):
        log(f"No prepared data in {data_dir}; preparing it now.")
        data.prepare(data_dir, log=log)

    if resuming:
        log("Found an earlier run: resuming with its saved settings "
            "(command-line training options are ignored; use --fresh to start again).")
        E, counts = data.load_prepared(data_dir, cfg["subset"])
        if args.extend_steps is not None or args.extend_hours is not None:
            _extend(run_dir, cfg, args, log)
    else:
        E, counts = data.load_prepared(data_dir, args.subset)
        cfg = _new_config(name, model, schedule, args, E, log)
        save_config(run_dir, cfg)

    sched = cfg.get("schedule")
    by_k = {int(k): v for k, v in cfg["step_seconds_by_k"].items()} if sched else None
    log("Settings: " + json.dumps({k: cfg.get(k) for k in (
        "V", "batch_size", "lr", "seed", "steps", "eval_every", "schedule", "l1_coef", "subset",
        "frequency_weighted")}))
    train_counts = counts if cfg["frequency_weighted"] else None

    _, history, finished = train(
        E, steps=cfg["steps"], batch_size=cfg["batch_size"], lr=cfg["lr"], seed=cfg["seed"],
        eval_every=cfg["eval_every"], model=cfg["model"], k=cfg.get("k"),
        l1_coef=cfg.get("l1_coef"), stages=cfg.get("k_stages"),
        counts=train_counts, device=cfg["device"], run_dir=run_dir,
        checkpoint_every=60 * cfg["checkpoint_minutes"], heartbeat_every=cfg["heartbeat_seconds"],
        extra_eta_seconds=ANALYSIS_EVALS * cfg["eval_seconds"], step_seconds_by_k=by_k,
        snapshot_steps=set(cfg.get("snapshot_steps", [])), log=log)
    if not finished:
        return None

    log("Training finished. Analysing ...")
    Et = torch.as_tensor(E, dtype=torch.float32)
    V, d = Et.shape
    # Features dead at initialisation are judged at the starting k; the final model at its last k.
    stages = cfg.get("k_stages")
    init = load_model_file(run_dir / INIT_FILE, cfg, d, V, k=stages[0][0] if stages else None)
    final = load_model_file(run_dir / MODEL_FILE, cfg, d, V, k=stages[-1][0] if stages else None)
    result = analyse(init, final, Et, token_weights(V, train_counts))
    result["training_hours"] = round(history[-1]["seconds"] / 3600, 2)
    (run_dir / ANALYSIS_FILE).write_text(json.dumps(result, indent=2))
    for key, value in result.items():
        log(f"  {key:>22}: {value:.4g}" if isinstance(value, float) else f"  {key:>22}: {value}")
    report(result, log)
    return result


def _extend(run_dir, cfg, args, log):
    """Raise the step count of an existing run so that training continues from its checkpoint.

    The new steps are a whole number of evaluation intervals. The earlier
    analysis, if any, is kept as analysis_step<N>.json. For a run with a k
    schedule, the part of the schedule after the checkpoint is stretched over
    the new length (stretch_stages). If the schedule has already reached its
    last k, that k simply continues.
    """
    if args.extend_steps is not None and args.extend_hours is not None:
        raise SystemExit("give --extend-steps or --extend-hours, not both")
    every = cfg["eval_every"]
    if args.extend_steps is not None:
        extra = args.extend_steps
    else:
        per_interval = every * cfg["step_seconds"] + cfg["eval_seconds"]
        extra = int(args.extend_hours * 3600 / per_interval) * every
    extra = max(-(-extra // every) * every, every)
    old = cfg["steps"]
    analysis = run_dir / ANALYSIS_FILE
    if analysis.exists():
        analysis.replace(run_dir / f"analysis_step{old}.json")
    cfg["steps"] = old + extra
    if cfg.get("k_stages"):
        ckpt = run_dir / CHECKPOINT_FILE
        done = torch.load(ckpt, map_location="cpu")["step"] if ckpt.exists() else 0
        cfg["k_stages"] = stretch_stages(cfg["k_stages"], done, cfg["steps"])
        log(f"The schedule after step {done:,} is stretched over the new length: "
            f"{describe_stages(cfg['k_stages'])}.")
    cfg.setdefault("extensions", []).append(
        {"from_step": old, "to_step": cfg["steps"], "when": time.strftime("%Y-%m-%d %H:%M")})
    new_snaps = pick_snapshot_steps(old, cfg["steps"], every, args.n_snapshots)
    cfg["snapshot_steps"] = cfg.get("snapshot_steps", []) + new_snaps
    save_config(run_dir, cfg)
    log(f"Extending the run from {old:,} to {cfg['steps']:,} steps "
        f"({extra // every} more evaluations, about "
        f"{fmt_duration(extra * cfg['step_seconds'] + extra // every * cfg['eval_seconds'])} "
        f"at the measured speed).")
    log(f"Snapshots will be kept at steps {', '.join(f'{s:,}' for s in new_snaps)}.")


def _new_config(name, model, schedule, args, E, log):
    V = E.shape[0]
    sched = None
    if schedule is not None:
        sched = {"shape": "stages", "stages": args.stages}
    cfg = {
        "experiment": name, "model": model, "l1_coef": getattr(args, "l1_coef", None),
        "k": schedule_start_k(sched) if sched else None,
        "schedule": sched, "data_dir": args.data_dir, "subset": args.subset, "V": V,
        "d": E.shape[1], "batch_size": args.batch_size, "lr": args.lr or default_lr(V),
        "seed": args.seed, "frequency_weighted": args.frequency_weighted, "hours": args.hours,
        "n_evals": args.n_evals, "checkpoint_minutes": args.checkpoint_minutes,
        "heartbeat_seconds": args.heartbeat_seconds, "device": args.device,
        "python": platform.python_version(), "torch": torch.__version__,
        "threads": torch.get_num_threads(), "started": time.strftime("%Y-%m-%d %H:%M"),
    }
    log(f"Timing {args.batch_size}-token steps on {V:,} tokens "
        f"({torch.get_num_threads()} threads) ...")
    if sched is None:
        step_s, eval_s = time_training(E, args.batch_size, model=model,
                                       l1_coef=cfg["l1_coef"], device=args.device)
        log(f"  {step_s:.2f} s per step, {eval_s:.0f} s per evaluation")
    else:
        by_k = {}
        for k in schedule_ks(sched):
            by_k[k], ev = time_training(E, args.batch_size, model="topk", k=k,
                                        device=args.device, evaluate=(k == 1))
            if ev is not None:
                eval_s = ev
            log(f"  k = {k}: {by_k[k]:.2f} s per step")
        cfg["step_seconds_by_k"] = {str(k): round(v, 3) for k, v in by_k.items()}
        step_s = mean_step_seconds(by_k, sched)
        log(f"  mean over the schedule: {step_s:.2f} s per step; {eval_s:.0f} s per evaluation")
    cfg.update(step_seconds=round(step_s, 3), eval_seconds=round(eval_s, 1))
    if args.steps is None:
        cfg["steps"], cfg["eval_every"] = plan_steps(
            args.hours, step_s, eval_s, args.n_evals, extra_evals=ANALYSIS_EVALS)
    else:
        cfg["steps"], cfg["eval_every"] = args.steps, max(args.steps // args.n_evals, 1)
    cfg["snapshot_steps"] = pick_snapshot_steps(0, cfg["steps"], cfg["eval_every"],
                                                args.n_snapshots)
    expected = cfg["steps"] * step_s + (args.n_evals + 1 + ANALYSIS_EVALS) * eval_s
    log(f"  {cfg['steps']:,} steps, evaluation every {cfg['eval_every']:,}, "
        f"expected {fmt_duration(expected)}.")
    if sched is not None:
        cfg["k_stages"] = k_stages(sched, cfg["steps"])
        log(f"  Schedule: {describe_stages(cfg['k_stages'])}.")
    log(f"  Snapshots will be kept at steps "
        f"{', '.join(f'{s:,}' for s in cfg['snapshot_steps'])}.")
    return cfg
