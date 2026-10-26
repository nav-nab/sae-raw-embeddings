import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / "experiments" / "exp6_hull_vertices" / "run.py"
spec = importlib.util.spec_from_file_location("exp6", path)
exp6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp6)


def test_own_direction_margin_matches_brute_force():
    E = np.random.default_rng(0).normal(size=(50, 6))
    got = exp6.own_direction_margin(E, chunk=7)
    for j in range(50):
        u = E[j] / np.linalg.norm(E[j])
        best = max(E[i] @ u for i in range(50) if i != j)
        assert np.isclose(got[j], (np.linalg.norm(E[j]) - best) / np.linalg.norm(E[j]), atol=1e-5)


def test_linear_programme_finds_vertices_and_rejects_interior_points():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(30, 5))
    interior = 0.3 * V[0] + 0.5 * V[1] + 0.2 * V[2]  # a convex combination
    E = np.vstack([V, interior])
    opt, _, _ = exp6.separate(E, len(E) - 1)
    assert opt <= exp6.TOL
    # A short vector can fail the own-direction test yet still be a vertex.
    F = np.array([[1.0, 0.0], [3.0, 3.0], [-1.0, 3.0], [-3.0, -1.0], [-1.0, -2.0]])
    assert exp6.own_direction_margin(F)[0] <= 0
    opt, margin, _ = exp6.separate(F, 0)
    assert opt > exp6.TOL and margin > 0
