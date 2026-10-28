"""Run several experiments one after another, unattended overnight.

Run from the repository root:

    caffeinate -i python scripts/run_overnight.py               # the outstanding runs
    caffeinate -i python scripts/run_overnight.py --plan night1 # an earlier night

Plans:

    remaining (default): the Experiment 7 extension (about 6 hours on the
        laptop), then Experiment 8 (about 5 hours). About 11 hours in all.
    night1 (2026-10-26): Experiments 7, 8 and 9 at 5,000 and 10,000 tokens.

Each experiment runs exactly as if started on its own, with its own results
folder, log and heartbeat. Finished runs are skipped, so a plan can be run
again safely: it resumes where it stopped.

Ctrl-C stops the experiment that is running, after saving a checkpoint, and
does not start the later ones. If one experiment fails with an error, the
error is printed and the next experiment still starts, so that one bug does
not waste the night. A summary at the end says what finished.
"""

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP7 = "experiments/exp7_topk_lr_steps/run.py"
EXP8 = "experiments/exp8_topk_sizes/run.py"
EXP9 = "experiments/exp9_relu_sizes/run.py"
PLANS = {
    "remaining": [
        ("Experiment 7 extension", EXP7, []),
        ("Experiment 8", EXP8, []),
    ],
    "night1": [
        ("Experiment 7", EXP7, []),
        ("Experiment 8", EXP8, []),
        ("Experiment 9 (5,000 and 10,000 tokens)", EXP9, ["--sizes", "5000", "10000"]),
    ],
}


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).parent.name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_plan(plan, extra_args=()):
    """Run each (name, script, args) in turn; returns 0 if all finished, else 1."""
    outcomes = []
    for name, path, args in plan:
        print(f"\n{'#' * 70}\n# {time.strftime('%Y-%m-%d %H:%M:%S')}  Starting {name}\n{'#' * 70}",
              flush=True)
        try:
            result = load(path).main(list(args) + list(extra_args))
        except Exception:
            traceback.print_exc()
            outcomes.append((name, "failed with an error (traceback above)"))
            continue
        if result is None:
            outcomes.append((name, "interrupted"))
            print(f"{name} was interrupted; the later experiments were not started. "
                  "Run this script again to resume.", flush=True)
            break
        outcomes.append((name, "finished"))
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}  Summary:")
    for name, outcome in outcomes:
        print(f"  {name}: {outcome}")
    for name, _, _ in plan[len(outcomes):]:
        print(f"  {name}: not started")
    all_done = len(outcomes) == len(plan) and all(o == "finished" for _, o in outcomes)
    return 0 if all_done else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", choices=sorted(PLANS), default="remaining")
    args, extra = p.parse_known_args(argv)
    return run_plan(PLANS[args.plan], extra)


if __name__ == "__main__":
    sys.exit(main())
