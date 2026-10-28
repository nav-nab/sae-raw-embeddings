import pytest
import torch

from sae_raw.model import ReLUSAE, Top1SAE, TopKSAE, build_model


def naive_forward(model, x):
    """Reference: Top1(ReLU(a)) with a dense code, using plain autograd."""
    a = x @ model.W_enc.T + model.b_enc
    r = torch.relu(a)
    vals, idx = r.max(dim=1)
    z = torch.zeros_like(r).scatter(1, idx.unsqueeze(1), vals.unsqueeze(1))
    return z @ model.W_dec


def test_init_is_unit_norm_decoder_transpose_encoder_zero_bias():
    model = Top1SAE(d=16, m=40, seed=0)
    assert torch.allclose(model.W_dec.norm(dim=1), torch.ones(40), atol=1e-6)
    assert torch.equal(model.W_enc, model.W_dec)
    assert model.W_enc.data_ptr() != model.W_dec.data_ptr()  # separate parameters
    assert torch.count_nonzero(model.b_enc) == 0


def test_init_is_deterministic_given_seed():
    a, b, c = Top1SAE(8, 20, seed=1), Top1SAE(8, 20, seed=1), Top1SAE(8, 20, seed=2)
    assert torch.equal(a.W_dec, b.W_dec)
    assert not torch.equal(a.W_dec, c.W_dec)


def test_exact_solution_gives_zero_loss_and_identity_assignment():
    torch.manual_seed(0)
    E = torch.randn(60, 16) * (1 + torch.rand(60, 1))  # varied norms
    model = Top1SAE(d=16, m=60, seed=0)
    model.set_exact_solution_(E)
    x_hat, idx, coef = model(E)
    assert torch.allclose(x_hat, E, atol=1e-5)
    assert torch.equal(idx, torch.arange(60))
    assert torch.allclose(coef, E.norm(dim=1), atol=1e-5)


def test_no_positive_preactivation_gives_zero_code():
    model = Top1SAE(d=8, m=20, seed=0)
    with torch.no_grad():
        model.b_enc.fill_(-1e6)
    x_hat, idx, coef = model(torch.randn(5, 8))
    assert torch.count_nonzero(coef) == 0
    assert torch.count_nonzero(x_hat) == 0


def test_ties_pick_first_index():
    model = Top1SAE(d=4, m=3, seed=0)
    with torch.no_grad():
        model.W_enc.copy_(torch.ones(3, 4))  # identical scores for every feature
    assert model.winners(torch.ones(2, 4)).tolist() == [0, 0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_naive_forward_and_gradients(seed):
    torch.manual_seed(seed)
    model = Top1SAE(d=12, m=30, seed=seed)
    with torch.no_grad():
        model.b_enc.normal_(0, 0.3)  # exercise the bias path and some negative winners
    x = torch.randn(25, 12)

    x_hat, _, _ = model(x)
    loss = ((x - x_hat) ** 2).sum(dim=1).mean()
    loss.backward()
    fast = [p.grad.clone() for p in (model.W_enc, model.W_dec, model.b_enc)]
    for p in model.parameters():
        p.grad = None

    ref_hat = naive_forward(model, x)
    ref_loss = ((x - ref_hat) ** 2).sum(dim=1).mean()
    ref_loss.backward()
    ref = [p.grad.clone() for p in (model.W_enc, model.W_dec, model.b_enc)]

    assert torch.allclose(x_hat, ref_hat, atol=1e-6)
    for g_fast, g_ref in zip(fast, ref):
        assert torch.allclose(g_fast, g_ref, atol=1e-6)


def test_only_winners_receive_gradient():
    model = Top1SAE(d=8, m=50, seed=0)
    x = torch.randn(4, 8)
    x_hat, idx, _ = model(x)
    ((x - x_hat) ** 2).sum().backward()
    touched = (model.W_dec.grad.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
    assert set(touched) <= set(idx.tolist())


def test_projection_and_normalisation():
    model = Top1SAE(d=8, m=50, seed=0)
    x = torch.randn(16, 8)
    x_hat, _, _ = model(x)
    ((x - x_hat) ** 2).sum().backward()
    model.project_out_parallel_grad_()
    parallel = (model.W_dec.grad * model.W_dec).sum(dim=1)
    assert torch.allclose(parallel, torch.zeros(50), atol=1e-6)

    with torch.no_grad():
        model.W_dec.mul_(torch.rand(50, 1) + 0.5)
    model.normalize_decoder_()
    assert torch.allclose(model.W_dec.norm(dim=1), torch.ones(50), atol=1e-6)


# ----------------------------------------------------------------------------
# Shared initialisation and the L1 ReLU model
# ----------------------------------------------------------------------------

def test_both_models_start_from_identical_parameters():
    a = build_model("top1", 8, 20, seed=3)
    b = build_model("l1", 8, 20, seed=3, l1_coef=0.1)
    for name in ("W_enc", "W_dec", "b_enc"):
        assert torch.equal(getattr(a, name), getattr(b, name))


def test_relu_loss_is_mse_plus_l1():
    torch.manual_seed(0)
    model = ReLUSAE(d=12, m=30, l1_coef=0.3, seed=0)
    x = torch.randn(25, 12)
    f = torch.relu(x @ model.W_enc.T + model.b_enc)
    x_hat = f @ model.W_dec
    mse = ((x - x_hat) ** 2).sum(dim=1).mean()
    loss, mse_out = model.loss(x)
    assert torch.allclose(mse_out, mse)
    assert torch.allclose(loss, mse + 0.3 * f.sum(dim=1).mean())


def test_relu_gradient_reaches_every_active_feature_and_no_inactive_one():
    torch.manual_seed(0)
    model = ReLUSAE(d=8, m=50, l1_coef=0.1, seed=0)
    x = torch.randn(4, 8)
    _, _, active = model.activity(x)
    model.loss(x)[0].backward()
    touched = model.W_enc.grad.abs().sum(dim=1) > 0
    assert torch.equal(touched, active)
    assert active.sum() > 1  # unlike Top1, many features get gradient from one batch


def test_activity_counts():
    torch.manual_seed(0)
    x = torch.randn(10, 8)
    top1 = Top1SAE(8, 30, seed=0)
    _, l0, active = top1.activity(x)
    _, idx, coef = top1(x)
    assert torch.equal(l0, (coef > 0).long())
    assert set(active.nonzero().flatten().tolist()) == set(idx[coef > 0].tolist())
    relu = ReLUSAE(8, 30, l1_coef=0.1, seed=0)
    _, l0, active = relu.activity(x)
    f = relu(x)[1]
    assert torch.equal(l0, (f > 0).sum(dim=1))
    assert torch.equal(active, (f > 0).any(dim=0))


def test_build_model_needs_l1_coef_for_l1():
    with pytest.raises(ValueError):
        build_model("l1", 8, 20)


# ----------------------------------------------------------------------------
# TopK with k > 1
# ----------------------------------------------------------------------------

def naive_topk_forward(model, x):
    """Reference: TopK(ReLU(a)) with a dense code, using plain autograd."""
    a = x @ model.W_enc.T + model.b_enc
    idx = a.detach().topk(model.k, dim=1).indices
    mask = torch.zeros_like(a).scatter(1, idx, 1.0)
    return (torch.relu(a) * mask) @ model.W_dec


@pytest.mark.parametrize("k", [1, 3, 7])
def test_topk_matches_naive_forward_and_gradients(k):
    torch.manual_seed(k)
    model = TopKSAE(d=12, m=30, k=k, seed=k)
    with torch.no_grad():
        model.b_enc.normal_(0, 0.3)  # some selected features have negative scores
    x = torch.randn(25, 12)
    x_hat, _, _ = model(x)
    ((x - x_hat) ** 2).sum(dim=1).mean().backward()
    fast = [p.grad.clone() for p in (model.W_enc, model.W_dec, model.b_enc)]
    for p in model.parameters():
        p.grad = None
    ref_hat = naive_topk_forward(model, x)
    ((x - ref_hat) ** 2).sum(dim=1).mean().backward()
    assert torch.allclose(x_hat, ref_hat, atol=1e-5)
    for g_fast, p in zip(fast, (model.W_enc, model.W_dec, model.b_enc)):
        assert torch.allclose(g_fast, p.grad, atol=1e-5)


def test_topk_with_k1_equals_top1():
    torch.manual_seed(0)
    x = torch.randn(20, 8)
    a, b = Top1SAE(8, 30, seed=0), TopKSAE(8, 30, k=1, seed=0)
    assert torch.allclose(a(x)[0], b(x)[0])


def test_topk_activity_counts_active_selected_features():
    torch.manual_seed(0)
    model = TopKSAE(8, 40, k=5, seed=0)
    with torch.no_grad():
        model.b_enc.normal_(0, 0.5)
    x = torch.randn(10, 8)
    _, l0, active = model.activity(x)
    _, idx, coef = model(x)
    assert torch.equal(l0, (coef > 0).sum(dim=1))
    assert l0.max() <= 5
    assert set(active.nonzero().flatten().tolist()) == set(idx[coef > 0].tolist())


def test_build_model_needs_k_for_topk():
    with pytest.raises(ValueError):
        build_model("topk", 8, 20)
    assert build_model("topk", 8, 20, k=4).k == 4


def test_alive_at_init_counts_match_full_pass():
    import importlib.util
    from pathlib import Path
    from sae_raw.evaluation import full_pass
    path = Path(__file__).resolve().parents[1] / "experiments" / "exp5_alive_at_init" / "run.py"
    spec = importlib.util.spec_from_file_location("exp5", path)
    exp5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp5)
    E = torch.randn(300, 16) + 0.5  # a shared mean component, as in the real table
    counts = exp5.alive_counts(E, seed=3, ks=[1, 2, 4, 8], chunk=64)
    for k in (1, 2, 4, 8):
        model = build_model("topk", 16, 300, seed=3, k=k)
        assert counts[k] == int(full_pass(model, E)[2].sum())


def test_relu_sae_with_zero_penalty_has_loss_equal_to_mse():
    model = build_model("l1", 16, 40, seed=0, l1_coef=0.0)
    x = torch.randn(8, 16)
    loss, mse = model.loss(x)
    assert torch.equal(loss, mse)
