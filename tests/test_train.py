import pytest
import numpy as np
import torch

from sae_raw import data
from sae_raw.evaluation import (changed_features, full_pass, recovery, recovery_summary,
                                revival, summarize)
from sae_raw.model import ReLUSAE, Top1SAE
from sae_raw.train import FrequencySampler, UniformSampler, default_lr, k_schedule, train


def test_sampler_is_uniform_and_in_range():
    draws = UniformSampler(7, seed=0)(140_000)
    assert draws.min() >= 0 and draws.max() < 7
    freq = torch.bincount(draws, minlength=7).double() / len(draws)
    assert torch.allclose(freq, torch.full((7,), 1 / 7, dtype=torch.float64), atol=0.005)


def test_sampler_is_deterministic_given_seed():
    a, b = UniformSampler(29, seed=3)(100), UniformSampler(29, seed=3)(100)
    assert torch.equal(a, b)


def test_frequency_sampler_matches_frequencies_and_skips_zero_counts():
    counts = np.array([0, 50, 0, 30, 20, 0, 0])  # trailing zero-count tokens
    draws = FrequencySampler(counts, seed=0)(200_000)
    freq = torch.bincount(draws, minlength=len(counts)).double() / len(draws)
    assert torch.allclose(freq, torch.tensor(counts / counts.sum()), atol=0.005)
    assert freq[[0, 2, 5, 6]].sum() == 0


@pytest.mark.parametrize("kind", ["top1", "l1"])
def test_full_pass_chunking_does_not_change_results(kind):
    torch.manual_seed(0)
    E = torch.randn(37, 8)
    model = Top1SAE(8, 37, seed=0) if kind == "top1" else ReLUSAE(8, 37, 0.1, seed=0)
    a = full_pass(model, E, chunk=5)
    b = full_pass(model, E, chunk=1000)
    for u, v in zip(a, b):
        assert torch.allclose(u.float(), v.float())


def test_weighted_summary_ignores_zero_weight_tokens():
    torch.manual_seed(0)
    E = torch.randn(40, 16) * (1 + torch.rand(40, 1))
    counts = torch.randint(0, 100, (40,)).double()
    counts[0] = 0
    p = counts / counts.sum()
    model = Top1SAE(16, 40, seed=0)
    model.set_exact_solution_(E)
    s = summarize(*full_pass(model, E, p), E, p)
    assert s["rel_loss"] < 1e-8
    assert s["n_tokens"] == s["n_alive"] == int((counts > 0).sum())


def test_summary_of_exact_solution():
    torch.manual_seed(0)
    E = torch.randn(40, 16) * (1 + torch.rand(40, 1))
    model = Top1SAE(16, 40, seed=0)
    model.set_exact_solution_(E)
    s = summarize(*full_pass(model, E), E, UniformSampler(40, 0).p)
    assert s["rel_loss"] < 1e-8
    assert s["n_alive"] == s["n_tokens"] == 40
    assert s["mean_l0"] == 1.0 and s["n_zero_code"] == 0


def test_recovery_of_exact_solution_and_of_a_permuted_copy():
    torch.manual_seed(0)
    E = torch.randn(50, 16) * (1 + torch.rand(50, 1))
    perm = torch.randperm(50)
    W = (E / E.norm(dim=1, keepdim=True))[perm]
    best_cos, best_feat, best_tok = recovery(W, E, chunk=7)
    assert torch.allclose(best_cos, torch.ones(50), atol=1e-5)
    assert torch.equal(perm[best_feat], torch.arange(50))
    r = recovery_summary(best_cos, best_feat, best_tok, thresholds=(0.99,))
    assert r["n_mutual"] == r["n_recovered_cos0.99"] == 50


def test_recovery_counts_a_shared_feature_once():
    E = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    W = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])  # one useful feature, one useless
    r = recovery_summary(*recovery(W, E), thresholds=(0.9,))
    assert r["n_mutual"] == 1  # tokens 0 and 1 share feature 0, which picks token 0


def test_changed_features_ignores_renormalisation_rounding():
    model = Top1SAE(16, 40, seed=0)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(1000):
        model.normalize_decoder_()
    assert not changed_features(init, model).any()
    with torch.no_grad():
        model.W_enc[3, 0] += 1e-4  # about one Adam step at the default learning rate
    assert changed_features(init, model).nonzero().flatten().tolist() == [3]


def test_revival_counts():
    dead = torch.tensor([True, True, True, False])
    changed = torch.tensor([True, False, False, True])
    assert revival(dead, changed) == {"n_dead_at_init": 3, "n_revived": 1,
                                      "revival_pct": 100 / 3}


def test_select_subset_picks_most_frequent():
    E = np.arange(10, dtype=np.float32).reshape(5, 2)
    counts = np.array([5, 50, 0, 20, 1])
    E_sub, counts_sub, ids = data.select_subset(E, counts, 3)
    assert ids.tolist() == [1, 3, 0]
    assert counts_sub.tolist() == [50, 20, 5]
    assert np.array_equal(E_sub, E[[1, 3, 0]])


def _tiny_problem():
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 16)).astype(np.float32) * (1 + rng.random((40, 1)))


def test_training_reduces_loss_and_keeps_unit_norm_decoder():
    E = _tiny_problem()
    model, history, _ = train(E, steps=400, batch_size=64, lr=1e-2, seed=0,
                           eval_every=100, log=lambda *_: None)
    assert [h["step"] for h in history] == [0, 100, 200, 300, 400]
    assert history[-1]["rel_loss"] < 0.5 * history[0]["rel_loss"]
    assert torch.allclose(model.W_dec.norm(dim=1), torch.ones(40), atol=1e-5)


def test_training_is_deterministic_given_seed():
    E = _tiny_problem()
    kwargs = dict(steps=50, batch_size=32, lr=1e-2, eval_every=50, log=lambda *_: None)
    a, _, _ = train(E, seed=1, **kwargs)
    b, _, _ = train(E, seed=1, **kwargs)
    assert torch.equal(a.W_dec, b.W_dec) and torch.equal(a.W_enc, b.W_enc)


def test_run_dir_saves_initial_parameters_and_revival_is_logged(tmp_path):
    E = _tiny_problem()
    model, history, finished = train(E, steps=50, batch_size=32, lr=1e-2, seed=0,
                                     eval_every=50, run_dir=tmp_path, log=lambda *_: None)
    assert finished
    for name in ("init.pt", "checkpoint.pt", "model.pt", "eval_log.csv"):
        assert (tmp_path / name).exists()
    saved = torch.load(tmp_path / "init.pt")
    assert torch.equal(saved["W_dec"], Top1SAE(16, 40, seed=0).W_dec)
    changed = changed_features(saved, model)
    assert changed.any() and not changed.all()  # some features never won a batch
    first, last = history[0], history[-1]
    assert first["n_dead_at_init"] == 40 - first["n_alive"] and first["n_revived"] == 0
    assert last["n_dead_at_init"] == first["n_dead_at_init"]
    assert torch.equal(model.b_enc[~changed], saved["b_enc"][~changed])


@pytest.mark.parametrize("kind", ["top1", "l1"])
def test_resuming_gives_the_same_result_as_an_uninterrupted_run(tmp_path, kind):
    E = _tiny_problem()
    kw = dict(batch_size=32, lr=1e-2, seed=0, eval_every=10, model=kind, l1_coef=0.05,
              log=lambda *_: None)
    ref, ref_hist, _ = train(E, steps=30, **kw)
    _, _, finished = train(E, steps=17, run_dir=tmp_path, **kw)  # stops part-way
    assert finished  # a shorter run, which the second call extends
    res, hist, finished = train(E, steps=30, run_dir=tmp_path, **kw)
    assert finished
    for name in ("W_enc", "W_dec", "b_enc"):
        assert torch.equal(getattr(ref, name), getattr(res, name))
    assert [h["step"] for h in hist] == [0, 10, 17, 20, 30]  # the early stop adds one eval
    assert hist[-1]["mse"] == ref_hist[-1]["mse"]


def test_ctrl_c_stops_after_the_current_step_and_saves(tmp_path):
    import os
    import signal
    E = _tiny_problem()

    def log(msg):
        if msg.startswith("step 5/"):
            os.kill(os.getpid(), signal.SIGINT)

    _, _, finished = train(E, steps=100, batch_size=32, lr=1e-2, seed=0, eval_every=50,
                           run_dir=tmp_path, heartbeat_every=0, log=log)
    assert not finished
    assert torch.load(tmp_path / "checkpoint.pt")["step"] == 5
    _, hist, finished = train(E, steps=100, batch_size=32, lr=1e-2, seed=0, eval_every=50,
                              run_dir=tmp_path, log=lambda *_: None)
    assert finished and hist[-1]["step"] == 100


def test_heartbeat_reports_progress_and_eta(tmp_path):
    lines = []
    train(_tiny_problem(), steps=3, batch_size=8, lr=1e-2, seed=0, eval_every=10,
          heartbeat_every=0, log=lines.append)
    beats = [l for l in lines if l.startswith("step ")]
    assert len(beats) == 3 and "ETA" in beats[0] and "revived" in beats[0]


def test_l1_training_reduces_loss_and_activates_many_features():
    E = _tiny_problem()
    model, history, _ = train(E, steps=300, batch_size=64, lr=1e-2, seed=0, eval_every=100,
                              model="l1", l1_coef=0.05, log=lambda *_: None)
    assert isinstance(model, ReLUSAE)
    assert history[-1]["rel_loss"] < 0.5 * history[0]["rel_loss"]
    assert history[0]["mean_l0"] > 1
    assert torch.allclose(model.W_dec.norm(dim=1), torch.ones(40), atol=1e-5)


def test_frequency_weighted_training_runs_and_never_samples_zero_count_tokens():
    E = _tiny_problem()
    counts = np.ones(40, dtype=np.int64)
    counts[:10] = 0
    model, history, _ = train(E, steps=100, batch_size=32, lr=1e-2, seed=0, eval_every=100,
                              counts=counts, log=lambda *_: None)
    assert history[-1]["n_tokens"] == 30
    assert history[-1]["rel_loss"] < history[0]["rel_loss"]


def test_default_lr_follows_scaling_rule():
    assert np.isclose(default_lr(2 ** 14), 2e-4)
    assert np.isclose(default_lr(50_257), 1.14e-4, rtol=0.01)


def test_topk_training_reduces_loss_and_resumes(tmp_path):
    E = _tiny_problem()
    kw = dict(batch_size=32, lr=1e-2, seed=0, eval_every=10, model="topk", k=4,
              log=lambda *_: None)
    ref, hist, _ = train(E, steps=30, **kw)
    assert hist[-1]["rel_loss"] < hist[0]["rel_loss"] and hist[-1]["mean_l0"] <= 4
    train(E, steps=15, run_dir=tmp_path, **kw)
    res, _, _ = train(E, steps=30, run_dir=tmp_path, **kw)
    assert torch.equal(ref.W_dec, res.W_dec)


def test_k_schedule_halving_and_linear():
    ks = [k_schedule(s, 100, 32, 0.5, "halving") for s in range(1, 101)]
    assert ks[:10] == [32] * 10 and ks[10:20] == [16] * 10 and ks[40:50] == [2] * 10
    assert set(ks[50:]) == {1}
    lin = [k_schedule(s, 100, 32, 0.5, "linear") for s in range(1, 101)]
    assert lin[0] == 32 and lin[25] == 16 and set(lin[50:]) == {1}
    assert all(a >= b for a, b in zip(lin, lin[1:]))


def test_scheduled_training_reaches_k1_and_resumes_identically(tmp_path):
    E = _tiny_problem()
    sched = dict(k0=8, end_frac=0.5, shape="halving")
    kw = dict(batch_size=32, lr=1e-2, seed=0, eval_every=10, model="topk", schedule=sched,
              log=lambda *_: None)
    ref, hist, _ = train(E, steps=40, **kw)
    assert [h["k"] for h in hist] == [8, 4, 2, 1, 1] and hist[-1]["mean_l0"] <= 1
    _, _, done = train(E, steps=40, run_dir=tmp_path, heartbeat_every=0,
                       **{**kw, "log": _stop_after(13)})
    assert not done
    res, _, done = train(E, steps=40, run_dir=tmp_path, **kw)
    assert done and torch.equal(ref.W_dec, res.W_dec) and torch.equal(ref.W_enc, res.W_enc)


def _stop_after(step):
    import os
    import signal

    def log(msg):
        if msg.startswith(f"step {step}/"):
            os.kill(os.getpid(), signal.SIGINT)
    return log
