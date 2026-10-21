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
