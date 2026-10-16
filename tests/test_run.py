import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sae_raw import data
from sae_raw.run import plan_steps

REPO = Path(__file__).resolve().parents[1]


def test_plan_steps_fits_the_budget():
    steps, every = plan_steps(hours=10, step_s=2.62, eval_s=23, n_evals=20, extra_evals=3)
    assert steps == 20 * every
    assert steps * 2.62 + 24 * 23 <= 10 * 3600
    assert (steps + every) * 2.62 + 24 * 23 > 10 * 3600  # one more block would not fit


def _fake_data(path):
    rng = np.random.default_rng(0)
    E = (rng.normal(size=(60, 16)) * (1 + rng.random((60, 1)))).astype(np.float32)
    data.save_prepared(path, E, rng.integers(0, 100, size=60), {"note": "test"})


def _run_exp1(*args):
    return subprocess.run([sys.executable, str(REPO / "experiments" / "exp1_top1" / "run.py"),
                           *args], capture_output=True, text=True, cwd=REPO, timeout=300)


def test_experiment_1_script_end_to_end_and_resume(tmp_path):
    _fake_data(tmp_path / "data")
    common = ["--data-dir", str(tmp_path / "data"), "--run-dir", str(tmp_path / "run"),
              "--batch-size", "16", "--n-evals", "2"]
    r = _run_exp1(*common, "--steps", "40", "--heartbeat-seconds", "0")
    assert r.returncode == 0, r.stderr
    run = tmp_path / "run"
    for name in ("config.json", "init.pt", "checkpoint.pt", "model.pt", "eval_log.csv",
                 "analysis.json", "run.log"):
        assert (run / name).exists(), name
    result = json.loads((run / "analysis.json").read_text())
    assert 0 <= result["n_revived"] <= result["n_dead_at_init"]
    log = (run / "run.log").read_text()
    assert "ETA" in log and "Result:" in log

    # A second invocation resumes from the saved settings (here: already finished).
    r = _run_exp1(*common, "--steps", "999")
    assert r.returncode == 0, r.stderr
    assert "resuming with its saved settings" in r.stdout
    assert json.loads((run / "config.json").read_text())["steps"] == 40


def test_experiment_1_frequency_toggle(tmp_path):
    _fake_data(tmp_path / "data")
    r = _run_exp1("--data-dir", str(tmp_path / "data"), "--run-dir", str(tmp_path / "run"),
                  "--batch-size", "16", "--n-evals", "1", "--steps", "10", "--frequency-weighted")
    assert r.returncode == 0, r.stderr
    counts = np.load(tmp_path / "data" / "token_counts.npy")
    result = json.loads((tmp_path / "run" / "analysis.json").read_text())
    assert result["n_tokens"] == int((counts > 0).sum())


def _run_exp3(*args):
    return subprocess.run([sys.executable, str(REPO / "experiments" / "exp3_k_annealing" / "run.py"),
                           *args], capture_output=True, text=True, cwd=REPO, timeout=300)


def test_experiment_3_script_end_to_end_with_schedule_and_resume(tmp_path):
    _fake_data(tmp_path / "data")
    common = ["--data-dir", str(tmp_path / "data"), "--run-dir", str(tmp_path / "run"),
              "--batch-size", "16", "--n-evals", "4", "--k0", "4"]
    r = _run_exp3(*common, "--steps", "40", "--heartbeat-seconds", "0")
    assert r.returncode == 0, r.stderr
    run = tmp_path / "run"
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["schedule"] == {"k0": 4, "end_frac": 0.8, "shape": "halving"}
    assert set(cfg["step_seconds_by_k"]) == {"4", "2", "1"}
    log = (run / "run.log").read_text()
    assert "k=4" in log and "k=1" in log and "Result:" in log
    result = json.loads((run / "analysis.json").read_text())
    assert result["mean_l0"] <= 1  # the final model is Top1
    rows = (run / "eval_log.csv").read_text().splitlines()
    assert rows[1].split(",")[2] == "4" and rows[-1].split(",")[2] == "1"  # k column
    r = _run_exp3(*common)
    assert r.returncode == 0 and "resuming with its saved settings" in r.stdout


def test_schedule_helpers():
    from sae_raw.run import mean_step_seconds, schedule_ks
    sched = {"k0": 32, "end_frac": 0.8, "shape": "halving"}
    assert schedule_ks(sched) == [32, 16, 8, 4, 2, 1]
    by_k = {32: 5.0, 16: 3.0, 8: 2.0, 4: 1.5, 2: 1.2, 1: 1.0}
    expected = 0.16 * (5 + 3 + 2 + 1.5 + 1.2) + 0.2 * 1.0
    assert abs(mean_step_seconds(by_k, sched) - expected) < 0.01  # stage edges round
