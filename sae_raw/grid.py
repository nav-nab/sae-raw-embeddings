"""Grid searches on a subvocabulary: one short training run per combination of settings.

A cell is a dict with a name, a batch size and a learning rate factor (a
multiple of the Gao rule). An L1 ReLU cell (Experiment 4) has an l1_coef and
trains to the grid's token count. A TopK cell (Experiment 7) has model "topk",
a k schedule (as for k_schedule) and its own steps and eval_every.

run_grid() runs every cell of a grid in turn, each in its own folder under the
grid directory, with the resumable train() loop. One command runs the whole
grid:

1. it prepares the data if it is missing and takes the subvocabulary;
2. it times a few steps at each batch size, to estimate the time per cell;
3. it trains each cell to the same number of training tokens, with a
   full-vocabulary evaluation (including decoder recovery) every half
   "million" tokens. A million here is 2^20 tokens, so that the measurement
   points fall on whole steps at every batch size;
4. after each cell it writes the cell's result.json, rewrites summary.csv for
   the whole grid, and deletes the cell's checkpoint and initial parameters.

The learning rate is constant, so nothing in a run depends on its planned
length. A run to 8 million tokens therefore passes through the same states as
shorter runs would, and the evaluations along the way measure the effect of
training length at no extra cost.

Ctrl-C finishes the current step, saves a checkpoint and stops. Running the same
command again skips the finished cells and resumes the interrupted one. The
heartbeat's ETA covers the rest of the whole grid. Everything is logged to the
console and to <grid-dir>/run.log.
"""

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

from sae_raw import data
from sae_raw.evaluation import recovery_fields
from sae_raw.run import mean_step_seconds, schedule_ks, setup_logging, time_training
from sae_raw.train import (CHECKPOINT_FILE, INIT_FILE, default_lr, fmt_duration, train)

TOKEN_UNIT = 2 ** 20  # one "million" tokens
GRID_FILE = "grid.json"
RESULT_FILE = "result.json"
SUMMARY_FILE = "summary.csv"
SUMMARY_FIELDS = ("cell", "l1_coef", "lr_factor", "lr", "batch_size", "tokens_m", "step",
                  "n_alive", "mean_l0", "rel_loss", "n_recovered_cos0.9",
                  "n_recovered_cos0.99")
TOPK_SUMMARY_FIELDS = ("cell", "lr_factor", "lr", "batch_size", "steps", "step", "k",
                       "n_alive", "n_top1", "mean_l0", "rel_loss", "n_recovered_cos0.9",
                       "n_recovered_cos0.99")


def make_cells(l1_coefs, lr_factors, batch_sizes):
    """Every combination of the given values, as a list of dicts with a name each."""
    cells = []
    for lam, lrf, b in itertools.product(l1_coefs, lr_factors, batch_sizes):
        cells.append({"name": f"lam{lam:g}_lr{lrf:g}_b{b}", "l1_coef": lam,
                      "lr_factor": lrf, "batch_size": b})
    return cells


def is_topk(cell):
    return cell.get("model", "l1") == "topk"


def cell_steps(cell, max_tokens_m, eval_tokens_m):
    """(steps, eval_every) for a cell: its own for a TopK cell; otherwise training to
    max_tokens_m million tokens, with an evaluation every eval_tokens_m million tokens."""
    if is_topk(cell):
        return cell["steps"], cell["eval_every"]
    b = cell["batch_size"]
    steps = int(max_tokens_m * TOKEN_UNIT) // b
    every = int(eval_tokens_m * TOKEN_UNIT) // b
    if steps * b != max_tokens_m * TOKEN_UNIT or every * b != eval_tokens_m * TOKEN_UNIT:
        raise ValueError(f"batch size {b} does not divide the token counts evenly")
    return steps, every


def estimate_seconds(cell, grid):
    """Expected time for one cell, from the timings measured at the start of the grid."""
    steps, every = cell_steps(cell, grid["max_tokens_m"], grid["eval_tokens_m"])
    b = str(cell["batch_size"])
    if is_topk(cell):
        by_k = {int(k): s for k, s in grid["step_seconds_by_k"][b].items()}
        step_s = mean_step_seconds(by_k, cell["schedule"])
    else:
        step_s = grid["step_seconds"][b]
    return steps * step_s + (steps // every + 1) * grid["eval_seconds"]


def _parser(description, default_dir):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", default=default_dir)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--subset", type=int, default=2000,
                   help="train on the N most frequent tokens, with m = N")
    p.add_argument("--max-tokens-m", type=float, default=8,
                   help="training length of each cell, in millions (2^20) of tokens")
    p.add_argument("--eval-tokens-m", type=float, default=0.5,
                   help="evaluate every this many millions of tokens")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-minutes", type=float, default=10.0)
    p.add_argument("--heartbeat-seconds", type=float, default=60.0)
    p.add_argument("--device", default="cpu")
    return p


def run_grid(name, description, default_dir, cells, argv=None, eval_extra=recovery_fields):
    """Run every cell of a grid (see the module docstring).

    eval_extra(model, E) adds fields to every evaluation; by default decoder recovery.
    """
    args = _parser(description, default_dir).parse_args(argv)
    grid_dir = Path(args.grid_dir)
    log = setup_logging(grid_dir)
    log("=" * 70)
    log(f"{name}: grid directory {grid_dir}")

    grid_path = grid_dir / GRID_FILE
    if grid_path.exists():
        grid = json.loads(grid_path.read_text())
        log("Found an earlier grid: resuming with its saved settings "
            "(command-line options are ignored).")
    else:
        grid = None
    data_dir = grid["data_dir"] if grid else args.data_dir
    if not data.is_prepared(data_dir):
        log(f"No prepared data in {data_dir}; preparing it now.")
        data.prepare(data_dir, log=log)
    E, _ = data.load_prepared(data_dir, grid["subset"] if grid else args.subset)
    V = E.shape[0]

    if grid is None:
        grid = {"experiment": name, "data_dir": args.data_dir, "subset": args.subset, "V": V,
                "seed": args.seed, "max_tokens_m": args.max_tokens_m,
                "eval_tokens_m": args.eval_tokens_m, "base_lr": default_lr(V),
                "checkpoint_minutes": args.checkpoint_minutes,
                "heartbeat_seconds": args.heartbeat_seconds, "device": args.device,
                "cells": cells, "started": time.strftime("%Y-%m-%d %H:%M")}
        for cell in cells:
            cell_steps(cell, args.max_tokens_m, args.eval_tokens_m)  # check the token counts
        log(f"Timing steps on {V:,} tokens ...")
        grid["step_seconds"], grid["step_seconds_by_k"] = {}, {}
        eval_s = None
        for b in sorted({c["batch_size"] for c in cells if not is_topk(c)}):
            l1 = next(c["l1_coef"] for c in cells if not is_topk(c))
            step_s, ev = time_training(E, b, model="l1", l1_coef=l1,
                                       device=args.device, evaluate=eval_s is None)
            eval_s = ev if ev is not None else eval_s
            grid["step_seconds"][str(b)] = round(step_s, 4)
            log(f"  batch size {b}: {step_s:.3f} s per step")
        for b in sorted({c["batch_size"] for c in cells if is_topk(c)}):
            ks = sorted({k for c in cells if is_topk(c) and c["batch_size"] == b
                         for k in schedule_ks(c["schedule"])}, reverse=True)
            grid["step_seconds_by_k"][str(b)] = {}
            for k in ks:
                step_s, ev = time_training(E, b, model="topk", k=k, device=args.device,
                                           evaluate=eval_s is None)
                eval_s = ev if ev is not None else eval_s
                grid["step_seconds_by_k"][str(b)][str(k)] = round(step_s, 4)
            log(f"  batch size {b}, TopK: " + ", ".join(
                f"k={k} {s:.3f} s" for k, s in grid["step_seconds_by_k"][str(b)].items())
                + " per step")
        grid["eval_seconds"] = round(eval_s, 2)
        grid_dir.mkdir(parents=True, exist_ok=True)
        grid_path.write_text(json.dumps(grid, indent=2))

    cells = grid["cells"]
    done = lambda c: (grid_dir / c["name"] / RESULT_FILE).exists()
    total = sum(estimate_seconds(c, grid) for c in cells)
    left = sum(estimate_seconds(c, grid) for c in cells if not done(c))
    length = ("each with its own number of steps" if all(map(is_topk, cells))
              else f"each to {grid['max_tokens_m']:g} million tokens")
    log(f"{len(cells)} cells, {length}; "
        f"{sum(map(done, cells))} already done. Expected time: {fmt_duration(total)} "
        f"for the whole grid, {fmt_duration(left)} still to run.")

    for i, cell in enumerate(cells, 1):
        if done(cell):
            continue
        cell_dir = grid_dir / cell["name"]
        steps, every = cell_steps(cell, grid["max_tokens_m"], grid["eval_tokens_m"])
        lr = grid["base_lr"] * cell["lr_factor"]
        later = sum(estimate_seconds(c, grid) for c in cells[i:] if not done(c))
        prefix = f"[{i}/{len(cells)} {cell['name']}] "
        what = "TopK, k annealed" if is_topk(cell) else f"lambda {cell['l1_coef']:g}"
        log(f"{prefix}{what}, learning rate {lr:.3g} "
            f"({cell['lr_factor']:g} x the Gao rule), batch size {cell['batch_size']}: "
            f"{steps:,} steps, evaluation every {every:,}.")
        model_kw = ({"model": "topk", "schedule": cell["schedule"]} if is_topk(cell)
                    else {"model": "l1", "l1_coef": cell["l1_coef"]})
        _, history, finished = train(
            E, steps=steps, batch_size=cell["batch_size"], lr=lr, seed=grid["seed"],
            eval_every=every, device=grid["device"],
            run_dir=cell_dir, checkpoint_every=60 * grid["checkpoint_minutes"],
            heartbeat_every=grid["heartbeat_seconds"], extra_eta_seconds=later,
            eval_extra=eval_extra, log=lambda msg: log(prefix + msg), **model_kw)
        if not finished:
            log("Stopped early. Run the same command again to resume.")
            return None
        rows = _cell_rows(cell, lr, history, grid)
        (cell_dir / RESULT_FILE).write_text(json.dumps(rows, indent=2))
        for name_ in (CHECKPOINT_FILE, INIT_FILE):
            (cell_dir / name_).unlink(missing_ok=True)
        _write_summary(grid_dir, cells)
        last = rows[-1]
        log(f"{prefix}done: {last['n_recovered_cos0.9']:,} tokens recovered at cosine > 0.9 "
            f"and {last['n_recovered_cos0.99']:,} at > 0.99, {last['n_alive']:,} features "
            f"alive, {last['mean_l0']:.3g} active per token.")

    _write_summary(grid_dir, cells)
    log(f"Grid finished. Summary in {grid_dir / SUMMARY_FILE}. At the end of each cell:")
    for cell in cells:
        last = json.loads((grid_dir / cell["name"] / RESULT_FILE).read_text())[-1]
        log(f"  {cell['name']:>24}: recovered {last['n_recovered_cos0.9']:>5,} (> 0.9), "
            f"{last['n_recovered_cos0.99']:>5,} (> 0.99); alive {last['n_alive']:>5,}; "
            f"active per token {last['mean_l0']:.3g}")
    return grid


def _cell_rows(cell, lr, history, grid):
    """Summary rows of a finished cell.

    A TopK cell gets one row per evaluation. An L1 cell gets one per evaluation at a
    doubling of the token count, and one at the end.
    """
    if is_topk(cell):
        return [{"cell": cell["name"], "lr_factor": cell["lr_factor"], "lr": lr,
                 "batch_size": cell["batch_size"], "steps": cell["steps"],
                 **{k: rec.get(k) for k in TOPK_SUMMARY_FIELDS[5:]}} for rec in history]
    rows = []
    for rec in history:
        tokens_m = rec["step"] * cell["batch_size"] / TOKEN_UNIT
        if tokens_m == 0:
            continue
        ratio = tokens_m / grid["eval_tokens_m"]
        doubling = ratio.is_integer() and _is_power_of_two(int(ratio))
        if doubling or tokens_m == grid["max_tokens_m"]:
            rows.append({"cell": cell["name"], "l1_coef": cell["l1_coef"],
                         "lr_factor": cell["lr_factor"], "lr": lr,
                         "batch_size": cell["batch_size"], "tokens_m": tokens_m,
                         **{k: rec[k] for k in SUMMARY_FIELDS[6:]}})
    return rows


def _is_power_of_two(n):
    return n > 0 and n & (n - 1) == 0


def _write_summary(grid_dir, cells):
    rows = []
    for cell in cells:
        path = grid_dir / cell["name"] / RESULT_FILE
        if path.exists():
            rows.extend(json.loads(path.read_text()))
    fields = TOPK_SUMMARY_FIELDS if any(map(is_topk, cells)) else SUMMARY_FIELDS
    with open(grid_dir / SUMMARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
