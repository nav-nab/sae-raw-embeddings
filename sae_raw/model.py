"""Sparse autoencoders on the embedding table: Top1 (Experiment 1) and L1 ReLU (Experiment 2).

Conventions shared by both models
---------------------------------
Feature j has an encoder row W_enc[j], an encoder bias b_enc[j] and a decoder
row W_dec[j]. Both matrices have shape [m, d]. Decoder rows have unit norm and
there is no decoder bias, so the reconstruction is a non-negative combination
of decoder rows.

Initialisation uses no information about the data: decoder rows are isotropic
Gaussian directions normalised to unit length, the encoder starts as a copy of
the decoder, and the encoder bias is zero. Both matrices are then trained as
separate parameters. With the same seed, both models start from identical
parameters.

Training rule for the unit-norm decoder, to be applied in this order:

    loss, mse = model.loss(x)
    loss.backward()
    model.project_out_parallel_grad_()   # remove gradient along each decoder row
    optimizer.step()
    model.normalize_decoder_()           # renormalise rows after every step

Top1SAE, for a batch x of shape [B, d]:

    a     = x @ W_enc.T + b_enc            pre-activations, [B, m]
    j*    = argmax_j a                     winning feature (ties: first index)
    coef  = relu(a[j*])                    zero if no pre-activation is positive
    x_hat = coef * W_dec[j*]
    loss  = |x - x_hat|^2                  (mean over the batch)

Only the winner's parameters receive gradient. The winner is chosen without
autograd and its score recomputed from the gathered encoder row, which gives
the same gradients as the dense version (see tests) without a [B, m] gradient.

ReLUSAE, for a batch x of shape [B, d]:

    f     = relu(x @ W_enc.T + b_enc)      code, [B, m]
    x_hat = f @ W_dec
    loss  = |x - x_hat|^2 + l1_coef * sum_j f_j     (mean over the batch)

Every feature that is active on some token in the batch receives gradient.
With unit-norm decoder rows, the penalty on f is the usual SAE L1 penalty.
"""

import torch
from torch import nn


class _UnitNormSAE(nn.Module):
    """Parameters, initialisation and decoder constraint shared by both models."""

    def __init__(self, d, m, seed=None):
        super().__init__()
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        W_dec = torch.randn(m, d, generator=gen)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.clone())
        self.b_enc = nn.Parameter(torch.zeros(m))

    @property
    def m(self):
        return self.W_dec.shape[0]

    @property
    def d(self):
        return self.W_dec.shape[1]

    def loss(self, x):
        """Return (training loss, mean squared reconstruction error) for a batch."""
        raise NotImplementedError

    @torch.no_grad()
    def activity(self, x):
        """Return (x_hat, l0, active) for a batch, without gradients.

        l0[b] is the number of active features for row b. active[j] is True if
        feature j is active on at least one row of the batch.
        """
        raise NotImplementedError

    @torch.no_grad()
    def normalize_decoder_(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-12))

    @torch.no_grad()
    def project_out_parallel_grad_(self):
        """Remove from the decoder gradient its component along each decoder row."""
        g = self.W_dec.grad
        g -= (g * self.W_dec).sum(dim=1, keepdim=True) * self.W_dec

    @torch.no_grad()
    def set_exact_solution_(self, E):
        """Set W_enc = W_dec = normalised rows of E (shape [m, d]) and b_enc = 0.

        For Top1SAE this is the known zero-loss solution. It is used only in
        tests, never as a starting point for training.
        """
        V = E.to(self.W_dec.dtype)
        V = V / V.norm(dim=1, keepdim=True)
        self.W_dec.copy_(V)
        self.W_enc.copy_(V)
        self.b_enc.zero_()


def _mse(x, x_hat):
    return ((x - x_hat) ** 2).sum(dim=1).mean()


class Top1SAE(_UnitNormSAE):
    kind = "top1"

    @torch.no_grad()
    def winners(self, x):
        """Index of the feature with the largest pre-activation for each row of x."""
        return (x @ self.W_enc.T + self.b_enc).argmax(dim=1)

    def forward(self, x):
        """Return (x_hat, winner index, coefficient)."""
        idx = self.winners(x)
        score = (x * self.W_enc[idx]).sum(dim=1) + self.b_enc[idx]
        coef = torch.relu(score)
        x_hat = coef.unsqueeze(1) * self.W_dec[idx]
        return x_hat, idx, coef

    def loss(self, x):
        x_hat, _, _ = self(x)
        mse = _mse(x, x_hat)
        return mse, mse

    @torch.no_grad()
    def activity(self, x):
        x_hat, idx, coef = self(x)
        on = coef > 0  # a zero coefficient gives the winner no gradient
        active = torch.zeros(self.m, dtype=torch.bool, device=x.device)
        active[idx[on]] = True
        return x_hat, on.long(), active


class ReLUSAE(_UnitNormSAE):
    kind = "l1"

    def __init__(self, d, m, l1_coef, seed=None):
        super().__init__(d, m, seed=seed)
        self.l1_coef = float(l1_coef)

    def forward(self, x):
        """Return (x_hat, code f)."""
        f = torch.relu(x @ self.W_enc.T + self.b_enc)
        return f @ self.W_dec, f

    def loss(self, x):
        x_hat, f = self(x)
        mse = _mse(x, x_hat)
        return mse + self.l1_coef * f.sum(dim=1).mean(), mse

    @torch.no_grad()
    def activity(self, x):
        x_hat, f = self(x)
        on = f > 0
        return x_hat, on.sum(dim=1), on.any(dim=0)


def build_model(kind, d, m, seed=None, l1_coef=None):
    """Construct a model by name: "top1" or "l1" (the latter needs l1_coef)."""
    if kind == "top1":
        return Top1SAE(d, m, seed=seed)
    if kind == "l1":
        if l1_coef is None:
            raise ValueError("the l1 model needs l1_coef")
        return ReLUSAE(d, m, l1_coef, seed=seed)
    raise ValueError(f"unknown model kind {kind!r}")
