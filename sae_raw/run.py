"""Helpers shared by the experiment scripts (experiments/<experiment>/run.py).

Each experiment script does everything for one experiment: data preparation
if needed, timing the machine to fit a time budget, resumable training with
checkpoints and a heartbeat, and the final analysis. This module holds the
parts that do not depend on the experiment.
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch

from sae_raw.model import build_model
from sae_raw.train import (CHECKPOINT_FILE, EVAL_LOG_FILE, INIT_FILE, MODEL_FILE,
                           FrequencySampler, Trainer, UniformSampler, default_lr)

CONFIG_FILE = "config.json"
ANALYSIS_FILE = "analysis.json"
LOG_FILE = "run.log"
RUN_FILES = (CONFIG_FILE, INIT_FILE, CHECKPOINT_FILE, MODEL_FILE, EVAL_LOG_FILE, ANALYSIS_FILE)


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


def time_training(E, batch_size, model="top1", l1_coef=None, n_steps=3, device="cpu"):
    """Seconds per training step and per full evaluation, on a throwaway model."""
    tr = Trainer(E, lr=default_lr(len(E)), seed=0, model=model, l1_coef=l1_coef, device=device)
    tr.train_step(batch_size)  # warm-up
    t0 = time.time()
    for _ in range(n_steps):
        tr.train_step(batch_size)
    step_s = (time.time() - t0) / n_steps
    t0 = time.time()
    tr.evaluate()
    return step_s, time.time() - t0


def plan_steps(hours, step_s, eval_s, n_evals, extra_evals=0):
    """Choose (steps, eval_every) so that training plus evaluations fit in the budget.

    steps is a multiple of eval_every, giving exactly n_evals evaluations
    after step 0. extra_evals reserves time for work after training,
    counted in units of one evaluation.
    """
    budget = hours * 3600 - (n_evals + 1 + extra_evals) * eval_s
    eval_every = max(int(budget / step_s) // n_evals, 1)
    return eval_every * n_evals, eval_every


def load_config(run_dir):
    return json.loads((Path(run_dir) / CONFIG_FILE).read_text())


def save_config(run_dir, cfg):
    (Path(run_dir) / CONFIG_FILE).write_text(json.dumps(cfg, indent=2))


def token_weights(V, counts=None):
    """Evaluation weights: uniform over V tokens, or proportional to counts if given."""
    return UniformSampler(V, 0).p if counts is None else FrequencySampler(counts, 0).p


def load_model_file(path, cfg, d, m):
    model = build_model(cfg["model"], d, m, l1_coef=cfg.get("l1_coef"))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model

