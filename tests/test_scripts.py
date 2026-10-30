"""Tests for the analysis helpers in scripts/ and the Experiment 5 power-law fit."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from sae_raw.model import ReLUSAE

REPO = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


code_spread = load("scripts/code_spread.py", "code_spread")
exp5 = load("experiments/exp5_alive_at_init/run.py", "exp5")


def test_power_law_fit_recovers_exact_law():
    V = [2000, 5000, 10000, 50000]
    a, b, rms = exp5.power_law_fit(V, [1.5 * v ** 0.4 for v in V])
    assert np.isclose(a, 1.5) and np.isclose(b, 0.4) and rms < 1e-9


def test_code_spread_counts_and_cutoff():
    # Two tokens in 2 dimensions; the encoder is set by hand so the codes are known.
    model = ReLUSAE(2, 3, 0.1)
    with torch.no_grad():
        model.W_enc.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.05]]))
        model.b_enc.zero_()
    E = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # Token 0 uses feature 0 only. Token 1 uses features 1 (1.0) and 2 (0.05).
    r = code_spread.spread(model, E)
    assert r["all"]["1"] == 0.5 and r["all"]["2"] == 0.5
    assert r["cut"]["1"] == 1.0  # 0.05 is below 10% of 1.0
    assert r["n_multi"] == 1 and np.isclose(r["second_median"], 0.05)
