import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sae_raw import data
from sae_raw.grid import TOKEN_UNIT, cell_steps, make_cells

REPO = Path(__file__).resolve().parents[1]


def test_make_cells_and_steps():
    cells = make_cells([0.1, 0.3], [1.0], [256, 4096])
    assert [c["name"] for c in cells] == ["lam0.1_lr1_b256", "lam0.1_lr1_b4096",
                                          "lam0.3_lr1_b256", "lam0.3_lr1_b4096"]
    assert cell_steps(cells[0], 8, 0.5) == (8 * TOKEN_UNIT // 256, TOKEN_UNIT // 2 // 256)
    assert cell_steps(cells[1], 8, 0.5) == (2048, 128)


def _fake_data(path):
    rng = np.random.default_rng(0)
    E = (rng.normal(size=(60, 16)) * (1 + rng.random((60, 1)))).astype(np.float32)
    data.save_prepared(path, E, rng.integers(1, 100, size=60), {"note": "test"})


def test_grid_script_end_to_end_and_resume(tmp_path):
    _fake_data(tmp_path / "data")
    # A tiny grid through the library entry point, with token counts small enough to run fast.
    script = tmp_path / "grid_run.py"
    script.write_text(
        "import sys\n"
        "from sae_raw.grid import make_cells, run_grid\n"
        "run_grid('test_grid', 'test', 'unused', make_cells([0.1, 0.3], [1.0], [16, 32]),\n"
        "         argv=sys.argv[1:])\n")
    unit = 32 / TOKEN_UNIT  # 32 tokens, in units of 2^20
    args = [sys.executable, str(script), "--data-dir", str(tmp_path / "data"),
            "--grid-dir", str(tmp_path / "grid"), "--subset", "60",
            "--max-tokens-m", repr(8 * unit), "--eval-tokens-m", repr(unit),
            "--heartbeat-seconds", "0"]
    r = subprocess.run(args, capture_output=True, text=True, cwd=REPO, timeout=300)
    assert r.returncode == 0, r.stderr
    grid = tmp_path / "grid"
    summary = (grid / "summary.csv").read_text().splitlines()
    assert len(summary) == 1 + 4 * 4  # 4 cells, measured at 1, 2, 4 and 8 units of tokens
    assert "n_recovered_cos0.99" in summary[0]
    cell = grid / "lam0.1_lr1_b16"
    assert (cell / "model.pt").exists() and not (cell / "checkpoint.pt").exists()
    rows = json.loads((cell / "result.json").read_text())
    assert [row["step"] for row in rows] == [2, 4, 8, 16]
    log = (grid / "run.log").read_text()
    assert "Grid finished" in log and "[4/4 lam0.3_lr1_b32]" in log
    # Running again skips every finished cell.
    r = subprocess.run(args, capture_output=True, text=True, cwd=REPO, timeout=300)
    assert r.returncode == 0 and "4 already done" in r.stdout


def test_topk_grid_end_to_end(tmp_path):
    _fake_data(tmp_path / "data")
    script = tmp_path / "topk_grid.py"
    exp7 = REPO / "experiments" / "exp7_topk_lr_steps" / "run.py"
    script.write_text(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('exp7', {str(exp7)!r})\n"
        "exp7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp7)\n"
        "from sae_raw.grid import run_grid\n"
        "sched = {'shape': 'stages', 'stages': [[8, 0.4], [2, 0.4], [1, 0.2]]}\n"
        "cells = exp7.make_topk_cells([(40, 1.0), (40, 2.0)], schedule=sched, batch_size=16,\n"
        "                             n_evals=10)\n"
        "run_grid('t', 't', 'unused', cells, argv=sys.argv[1:], eval_extra=exp7.eval_extra)\n")
    args = [sys.executable, str(script), "--data-dir", str(tmp_path / "data"),
            "--grid-dir", str(tmp_path / "grid"), "--subset", "60", "--heartbeat-seconds", "0"]
    r = subprocess.run(args, capture_output=True, text=True, cwd=REPO, timeout=300)
    assert r.returncode == 0, r.stderr
    summary = (tmp_path / "grid" / "summary.csv").read_text().splitlines()
    assert summary[0].startswith("cell,V,model,l1_coef,lr_factor,lr,batch_size,steps,step,k,"
                                 "n_alive,n_top1")
    assert len(summary) == 1 + 2 * 11  # 2 cells, evaluated at step 0 and 10 more times
    rows = json.loads((tmp_path / "grid" / "steps40_lr1" / "result.json").read_text())
    by_step = {row["step"]: row for row in rows}
    assert by_step[32]["k"] == 2 and by_step[36]["k"] == 1
    # At k = 1 every alive feature is some token's top feature, and vice versa.
    assert by_step[40]["n_top1"] == by_step[40]["n_alive"]


def _load(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_matched_schedule_starts_at_k0_and_halves_from_the_power_of_two_below():
    exp8 = _load("exp8", "experiments/exp8_topk_sizes/run.py")
    assert [k for k, _ in exp8.matched_schedule(42)["stages"]] == [42, 32, 16, 8, 4, 2, 1]
    assert [k for k, _ in exp8.matched_schedule(71)["stages"]] == [71, 64, 32, 16, 8, 4, 2, 1]
    t10 = _load("exp7", "experiments/exp7_topk_lr_steps/run.py").SCHEDULE
    assert exp8.matched_schedule(128) == t10
    assert abs(sum(f for _, f in exp8.matched_schedule(55)["stages"]) - 1) < 1e-12
    # 2V steps, so 1,024 epochs' worth at batch size 512 at every size.
    for V, k0 in exp8.K0_BY_SIZE.items():
        (cell,) = exp8.cells_for(V, k0)
        assert cell["steps"] == 2 * V and cell["steps"] * cell["batch_size"] / V == 1024


def test_run_grids_over_sizes_then_extend_a_cell(tmp_path):
    _fake_data(tmp_path / "data")
    exp9 = _load("exp9", "experiments/exp9_relu_sizes/run.py")
    script = tmp_path / "sizes.py"
    script.write_text(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('exp9', {str(REPO / 'experiments/exp9_relu_sizes/run.py')!r})\n"
        "exp9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp9)\n"
        "from sae_raw.grid import run_grids\n"
        "cells = {V: exp9.cells_for(V, {0.1: 1, 0.3: 1}, batch_size=8, n_evals=5) for V in (30, 60)}\n"
        "run_grids('t', 't', 'unused', cells, argv=sys.argv[1:], eval_extra=exp9.eval_extra)\n")
    base = [sys.executable, str(script), "--root", str(tmp_path / "runs"),
            "--data-dir", str(tmp_path / "data"), "--heartbeat-seconds", "0"]
    r = subprocess.run(base + ["--sizes", "30"], capture_output=True, text=True, cwd=REPO,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "runs" / "V60").exists()
    cell = tmp_path / "runs" / "V30" / "V30_lam0.1"
    assert (cell / "checkpoint.pt").exists() and (cell / "init.pt").exists()
    rows = json.loads((cell / "result.json").read_text())
    assert [row["step"] for row in rows] == [0, 6, 12, 18, 24, 30]
    assert 0 <= rows[-1]["share_l0_1"] <= 1 and rows[-1]["max_l0"] >= 1
    # Extend one cell to 45 steps: it trains on from step 30 and nothing else reruns.
    ext = base + ["--sizes", "30", "--extend", "V30_lam0.1", "45"]
    r = subprocess.run(ext, capture_output=True, text=True, cwd=REPO, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "Resuming from" in r.stdout and "1 already done" in r.stdout
    rows = json.loads((cell / "result.json").read_text())
    assert [row["step"] for row in rows][-4:] == [30, 36, 42, 45]
    assert rows[-1]["steps"] == 45
    # The same command again changes nothing.
    r = subprocess.run(ext, capture_output=True, text=True, cwd=REPO, timeout=300)
    assert r.returncode == 0 and "changes nothing" in r.stdout and "2 already done" in r.stdout
    # An unknown cell name is refused.
    r = subprocess.run(base + ["--sizes", "30", "--extend", "nope", "50"], capture_output=True,
                       text=True, cwd=REPO, timeout=300)
    assert r.returncode != 0 and "no cell named" in r.stderr


def test_new_cells_are_added_to_a_finished_grid_with_their_own_seeds(tmp_path):
    _fake_data(tmp_path / "data")
    exp7 = REPO / "experiments" / "exp7_topk_lr_steps" / "run.py"
    def script(spec):
        path = tmp_path / "grid7.py"
        path.write_text(
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('exp7', {str(exp7)!r})\n"
            "exp7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp7)\n"
            "from sae_raw.grid import run_grid\n"
            "sched = {'shape': 'stages', 'stages': [[8, 0.4], [2, 0.4], [1, 0.2]]}\n"
            f"cells = exp7.make_topk_cells({spec!r}, schedule=sched, batch_size=16, n_evals=4)\n"
            "run_grid('t', 't', 'unused', cells, argv=sys.argv[1:], eval_extra=exp7.eval_extra)\n")
        return [sys.executable, str(path), "--data-dir", str(tmp_path / "data"),
                "--grid-dir", str(tmp_path / "grid"), "--subset", "60",
                "--heartbeat-seconds", "0"]
    r = subprocess.run(script([(20, 1.0)]), capture_output=True, text=True, cwd=REPO,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(script([(20, 1.0, 0), (20, 1.0, 1), (40, 1.0, 0)]),
                       capture_output=True, text=True, cwd=REPO, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "Adding 2 new cells" in r.stdout and "1 already done" in r.stdout
    grid = tmp_path / "grid"
    seed0 = json.loads((grid / "steps20_lr1" / "result.json").read_text())
    seed1 = json.loads((grid / "steps20_lr1_seed1" / "result.json").read_text())
    assert seed1[-1]["seed"] == 1 and seed0[-1]["step"] == seed1[-1]["step"] == 20
    assert seed0[-1]["rel_loss"] != seed1[-1]["rel_loss"]  # a different seed, a different run
    summary = (grid / "summary.csv").read_text().splitlines()
    assert summary[0].endswith(",seed") and len(summary) == 1 + 5 + 5 + 5
    assert all(line.endswith(",0") for line in summary[1:6])  # old rows get the grid's seed
