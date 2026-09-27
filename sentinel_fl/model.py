"""Residual NumPy classifier and local client training."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .utils import rng_for


class ResidualModel:
    """Small NumPy classifier: a linear skip connection plus a one-hidden-layer tanh residual branch.

    Parameters are exposed as one flat vector partitioned into named blocks, which is what the
    byte codec and the federated protocol operate on."""

    def __init__(self, dim, hidden=32, seed=0):
        self.dim = int(dim)
        self.hidden = int(hidden)
        rng = rng_for("model", seed)
        self.params = {"skip": np.zeros(dim), "bias": np.zeros(1)}
        if hidden:
            self.params.update(
                W1=rng.normal(0, 1 / np.sqrt(dim), size=(dim, hidden)), b1=np.zeros(hidden), W2=np.zeros(hidden)
            )
        self.blocks = []
        lo = 0
        for name, v in self.params.items():
            self.blocks.append((name, lo, lo + v.size))
            lo += v.size
        self.size = lo

    def flat(self):
        return np.concatenate([a.ravel() for a in self.params.values()])

    def set_flat(self, v):
        v = np.asarray(v, float)
        if len(v) != self.size or not np.isfinite(v).all():
            raise ValueError("Bad model vector")
        for name, lo, hi in self.blocks:
            self.params[name] = v[lo:hi].reshape(self.params[name].shape).copy()

    def logits(self, X):
        z = X @ self.params["skip"] + self.params["bias"][0]
        if self.hidden:
            z = z + np.tanh(X @ self.params["W1"] + self.params["b1"]) @ self.params["W2"]
        return z

    def loss_gradient(self, X, y, cw, anchor=None, prox=0.0, l2=0.0):
        p = self.params
        z = self.logits(X)
        w = cw[np.asarray(y, int)]
        den = w.sum()
        loss = float(np.sum(w * (np.logaddexp(0, z) - y * z)) / den)
        e = w * (expit(z) - y) / den
        grad = {"skip": X.T @ e, "bias": np.array([e.sum()])}
        if self.hidden:
            h = np.tanh(X @ p["W1"] + p["b1"])
            u = e[:, None] * p["W2"] * (1 - h * h)
            grad.update(W1=X.T @ u, b1=u.sum(axis=0), W2=h.T @ e)
        g = np.concatenate([grad[k].ravel() for k in p])
        theta = self.flat()
        if l2:
            loss += 0.5 * l2 * (theta @ theta)
            g += l2 * theta
        if prox and anchor is not None:
            d = theta - anchor
            loss += 0.5 * prox * (d @ d)
            g += prox * d
        return (loss, g)

    def parameter_support(self, available):
        """Boolean mask of the parameters a client can update given its available inputs."""
        support = []
        for name in self.params:
            if name == "skip":
                support.append(np.asarray(available, bool))
            elif name == "W1":
                support.append(np.repeat(available, self.hidden))
            else:
                support.append(np.ones(self.params[name].size, bool))
        return np.concatenate(support)


def l2_clip(v, C):
    """Scale a vector so that its Euclidean norm is at most ``C``."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v * min(1.0, float(C) / max(n, 1e-30))


def train_local(model, base, X, y, cw, cfg, rng, prox=0.0, support=None):
    """Local proximal Adam-style training from the global parameters; returns ``(delta, mean_loss)``.

    Gradients are masked to the client's supported parameters and norm-clipped; no optimiser state
    persists between rounds."""
    model.set_flat(base)
    m = np.zeros_like(base)
    v = np.zeros_like(base)
    support = np.ones_like(base, bool) if support is None else np.asarray(support, bool)
    if not len(y):
        raise ValueError("Empty client")
    losses = []
    for step in range(1, cfg.local_steps + 1):
        ix = rng.choice(len(y), size=min(cfg.batch_size, len(y)), replace=False)
        loss, g = model.loss_gradient(X[ix], y[ix], cw, base, prox, cfg.weight_decay)
        g[~support] = 0.0
        g = l2_clip(g, 10.0)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        new = model.flat() - cfg.lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-08)
        model.set_flat(new)
        losses.append(loss)
    delta = model.flat() - base
    delta[~support] = 0.0
    return (delta, float(np.mean(losses)))
