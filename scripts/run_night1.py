"""Night 1: Experiments 7, 8 and 9 (at 5,000 and 10,000 tokens), one after another.

Run from the repository root:

    caffeinate -i python scripts/run_night1.py

It runs, in order,

    experiments/exp7_topk_lr_steps/run.py
    experiments/exp8_topk_sizes/run.py
    experiments/exp9_relu_sizes/run.py --sizes 5000 10000

each exactly as if started on its own, with its own results folder, log and
heartbeat. Expected time on the laptop: about 5.5 hours.

Ctrl-C stops the experiment that is running, after saving a checkpoint, and
does not start the later ones. Running this script again resumes where it
stopped: finished runs are skipped. If one experiment fails with an error, the
error is printed and the next experiment still starts, so that one bug does
not waste the night; a summary at the end says what finished.

Night 2 is a single command:

    caffeinate -i python experiments/exp9_relu_sizes/run.py --sizes 20000
"""

import importlib.util
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLAN = [
    ("Experiment 7", "experiments/exp7_topk_lr_steps/run.py", []),
    ("Experiment 8", "experiments/exp8_topk_sizes/run.py", []),
    ("Experiment 9 (5,000 and 10,000 tokens)", "experiments/exp9_relu_sizes/run.py",
     ["--sizes", "5000", "10000"]),
]


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).parent.name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    outcomes = []
    for name, path, args in PLAN:
        print(f"\n{'#' * 70}\n# {time.strftime('%Y-%m-%d %H:%M:%S')}  Starting {name}\n{'#' * 70}",
              flush=True)
        try:
            result = load(path).main(args + argv)
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
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}  Night 1 summary:")
    for name, outcome in outcomes:
        print(f"  {name}: {outcome}")
    for name, _, _ in PLAN[len(outcomes):]:
        print(f"  {name}: not started")
    return 0 if all(o == "finished" for _, o in outcomes) and len(outcomes) == len(PLAN) else 1


if __name__ == "__main__":
    sys.exit(main())
