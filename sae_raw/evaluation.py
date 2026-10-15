"""Evaluation of a trained SAE over the whole vocabulary.

Every function here works on all V tokens, never on sampled minibatches.
Weights p are the token probabilities used in training: uniform (1/V each) by
default, or Wikipedia counts normalised to sum 1 when frequency weighting is on.

A feature is alive if it is active on at least one token with positive weight.
For Top1 this means it wins the token with a positive coefficient. For the
ReLU model it means its activation is positive. A feature that is not alive
receives no gradient.
"""

import torch

# A feature whose parameters moved by more than this has received gradient.
# Repeated renormalisation of a unit-norm row changes it by rounding error only
# (around 1e-7), while a single Adam step moves it by about the learning rate.
CHANGE_ATOL = 1e-5


@torch.no_grad()
def full_pass(model, E, p=None, chunk=1024):
    """Run every row of E through the model.

    Returns (sq_err, l0, active). sq_err and l0 have length V: the squared
    reconstruction error and the number of active features of each token.
    active has length m and marks the features active on at least one token
    with positive weight (any token if p is None).
    """
    errs, l0s = [], []
    active = torch.zeros(model.m, dtype=torch.bool, device=E.device)
    for start in range(0, E.shape[0], chunk):
        x = E[start:start + chunk]
        x_hat, l0, act = model.activity(x)
        if p is not None:  # the active set counts tokens with positive weight only
            keep = p[start:start + chunk] > 0
            if not keep.all():
                act = model.activity(x[keep])[2] if keep.any() else torch.zeros_like(act)
        active |= act
        errs.append(((x - x_hat) ** 2).sum(dim=1))
        l0s.append(l0)
    return torch.cat(errs), torch.cat(l0s), active


@torch.no_grad()
def summarize(sq_err, l0, active, E, p):
    """Summary of a full pass, weighted by the token probabilities p.

    With uniform p this is the plain mean over all tokens. rel_loss is the
    weighted squared error divided by the weighted squared norm, so 1 means
    "no better than reconstructing zero". mean_l0 is the weighted mean number of
    active features per token. n_zero_code counts tokens (with positive weight)
    that have no active feature.
    """
    p = p.to(torch.float64)
    in_data = p > 0
    mse = (p * sq_err.to(torch.float64)).sum()
    norm_sq = (E.to(torch.float64) ** 2).sum(dim=1)
    return {
        "mse": float(mse),
        "rel_loss": float(mse / (p * norm_sq).sum()),
        "mean_l0": float((p * l0.to(torch.float64)).sum()),
        "n_tokens": int(in_data.sum()),
        "n_alive": int(active.sum()),
        "n_zero_code": int(((l0 == 0) & in_data).sum()),
    }


@torch.no_grad()
def changed_features(init_state, model, atol=CHANGE_ATOL):
    """Boolean mask over features whose parameters differ from init_state."""
    dev = model.W_dec.device
    changed = (model.b_enc - init_state["b_enc"].to(dev)).abs() > atol
    for name in ("W_enc", "W_dec"):
        delta = (getattr(model, name) - init_state[name].to(dev)).abs()
        changed |= delta.amax(dim=1) > atol
    return changed


def revival(dead_at_init, changed):
    """Counts of features dead at initialisation that have revived since.

    A feature revived if its parameters changed: a feature that is never
    active receives no gradient and stays at its initial values.
    """
    n_dead = int(dead_at_init.sum())
    n_revived = int((dead_at_init & changed).sum())
    return {
        "n_dead_at_init": n_dead,
        "n_revived": n_revived,
        "revival_pct": 100.0 * n_revived / n_dead if n_dead else float("nan"),
    }


@torch.no_grad()
def recovery(W_dec, E, chunk=1024):
    """Match decoder rows to normalised embeddings by cosine similarity.

    Returns (tok_best_cos, tok_best_feat, feat_best_tok): for each token, the
    best cosine and the feature achieving it; for each feature, its best token.
    """
    U = E / E.norm(dim=1, keepdim=True)
    D = W_dec / W_dec.norm(dim=1, keepdim=True)
    V, m = U.shape[0], D.shape[0]
    tok_best_cos = torch.empty(V, device=E.device)
    tok_best_feat = torch.empty(V, dtype=torch.long, device=E.device)
    feat_best_cos = torch.full((m,), -2.0, device=E.device)
    feat_best_tok = torch.zeros(m, dtype=torch.long, device=E.device)
    for start in range(0, V, chunk):
        S = U[start:start + chunk] @ D.T  # [chunk, m]
        c, f = S.max(dim=1)
        tok_best_cos[start:start + chunk], tok_best_feat[start:start + chunk] = c, f
        c, t = S.max(dim=0)
        better = c > feat_best_cos
        feat_best_cos[better] = c[better]
        feat_best_tok[better] = t[better] + start
    return tok_best_cos, tok_best_feat, feat_best_tok


def recovery_summary(tok_best_cos, tok_best_feat, feat_best_tok, thresholds=(0.9, 0.99)):
    """How many tokens are recovered one-to-one.

    Token i and feature j are mutual nearest neighbours if j is i's best
    feature and i is j's best token. A full one-to-one assignment over a
    V x m matrix is too large to compute, so mutual nearest neighbours stand in
    for it. A token counts as recovered at threshold t if it has a mutual
    nearest feature with cosine above t.
    """
    tokens = torch.arange(len(tok_best_feat), device=tok_best_feat.device)
    mutual = feat_best_tok[tok_best_feat] == tokens
    out = {"n_mutual": int(mutual.sum()),
           "median_best_cos": float(tok_best_cos.median())}
    for t in thresholds:
        out[f"n_recovered_cos{t}"] = int((mutual & (tok_best_cos > t)).sum())
    return out


@torch.no_grad()
def analyse(init_model, final_model, E, p, thresholds=(0.9, 0.99)):
    """Final report of a run: revival, loss summary and decoder recovery.

    Features dead at initialisation are recomputed from init_model, and a
    feature has revived if its parameters in final_model differ from those in
    init_model.
    """
    _, _, active_init = full_pass(init_model, E, p)
    sq_err, l0, active = full_pass(final_model, E, p)
    return {
        **revival(~active_init, changed_features(init_model.state_dict(), final_model)),
        **summarize(sq_err, l0, active, E, p),
        **recovery_summary(*recovery(final_model.W_dec, E), thresholds=thresholds),
    }
