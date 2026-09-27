"""Sketch-based screening, per-client payload audit and support-normalised aggregation."""

from __future__ import annotations

import math

import numpy as np

from .utils import rng_for


class CountSketch:
    """Seeded CountSketch projection used as a compact, linear fingerprint of a decoded update."""

    def __init__(self, dim, ds, seed):
        rng = rng_for("projection", seed)
        self.dim = dim
        self.ds = ds
        self.bucket = rng.integers(0, ds, dim)
        self.sign = rng.choice([-1.0, 1.0], dim)
        self.loads = np.bincount(self.bucket, minlength=ds)
        self.opnorm = float(np.sqrt(self.loads.max()))

    def __call__(self, x):
        return np.bincount(self.bucket, weights=self.sign * np.asarray(x), minlength=self.ds)

    def null_vector(self, rng):
        """Unit vector in the null space of the projection (used to simulate an evasive client)."""
        z = rng.normal(size=self.dim)
        s = self(z)
        z -= self.sign * s[self.bucket] / np.maximum(self.loads[self.bucket], 1)
        return z / max(np.linalg.norm(z), 1e-20)


def geometric_median(A, weights=None, iterations=60, tol=1e-07, return_status=False):
    """Weiszfeld geometric median with Vardi-Zhang handling of iterates that land on a data point.

    With ``return_status`` also returns whether it converged and the iterations used."""
    A = np.asarray(A, float)
    w = np.ones(len(A)) if weights is None else np.asarray(weights, float)
    center = np.average(A, axis=0, weights=w)
    converged = False
    used = 0
    for it in range(iterations):
        used = it + 1
        d = np.linalg.norm(A - center, axis=1)
        on_point = d < 1e-12
        if on_point.any():
            d = np.maximum(d, 1e-12)
        ww = w / d
        new = np.average(A, axis=0, weights=ww)
        shift = np.linalg.norm(new - center)
        center = new
        if shift < tol:
            converged = True
            break
    return (center, converged, used) if return_status else center


def screen(sketches, cfg, round_index):
    """Per-client screening of decoded-update sketches against a leave-one-out geometric-median consensus.

    A client is flagged on a reversed or far-off direction, or an excessive norm, and receives
    ``screen_min_weight``. Rounds without a usable consensus are reported as such, not silently passed."""
    ids = sorted(sketches)
    weights = {k: 1.0 for k in ids}
    diag = {}
    if len(ids) < cfg.screen_min_clients or round_index <= cfg.screen_warmup:
        return (weights, {k: dict(screened=False, flagged=False, cos=np.nan, weight=1.0, reason="warmup") for k in ids})
    mad_floor = float(getattr(cfg, "screen_mad_floor", 0.1))
    A = np.stack([sketches[k] for k in ids])
    norm = np.linalg.norm(A, axis=1)
    U = A / np.maximum(norm[:, None], 1e-12)
    for j, cid in enumerate(ids):
        others = np.arange(len(ids)) != j
        center, converged, _ = geometric_median(U[others], return_status=True)
        cn = np.linalg.norm(center)
        if cn < 0.1 or not converged:
            diag[cid] = dict(
                screened=False,
                flagged=False,
                cos=np.nan,
                weight=1.0,
                reason="no_consensus" if cn < 0.1 else "median_not_converged",
            )
            continue
        center /= cn
        cos = float(U[j] @ center)
        peer = U[others] @ center
        med = float(np.median(peer))
        scale = max(1.4826 * np.median(np.abs(peer - med)), mad_floor)
        zcos = (med - cos) / scale
        logs = np.log(np.maximum(norm, 1e-12))
        lm = float(np.median(logs[others]))
        ls = max(1.4826 * np.median(np.abs(logs[others] - lm)), 0.25)
        zn = (logs[j] - lm) / ls
        bad_dir = cos < cfg.screen_cos_floor or (zcos > cfg.screen_z and med - cos > cfg.screen_cos_gap)
        bad_norm = zn > cfg.screen_z and logs[j] - lm > cfg.screen_log_norm_gap
        flagged = bool(bad_dir or bad_norm)
        w = cfg.screen_min_weight if flagged else 1.0
        weights[cid] = w
        diag[cid] = dict(
            screened=True,
            flagged=flagged,
            cos=cos,
            z_cos=float(zcos),
            z_norm=float(zn),
            weight=w,
            reason="flagged" if flagged else "clean",
        )
    return (weights, diag)


def aggregate_updates(A, weights, supports):
    """Weighted mean of decoded updates, normalised per coordinate by the weight of clients that support it.

    A coordinate that a client cannot update (for example an unavailable sensor) therefore does not
    dilute the clients that can. Coordinates supported by nobody are zero.
    """
    A = np.asarray(A, float)
    weights = np.asarray(weights, float)
    supports = np.asarray(supports, bool)
    if not len(A):
        raise ValueError("No updates")
    numerator = (A * weights[:, None]).sum(axis=0)
    den = (supports * weights[:, None]).sum(axis=0)
    return np.divide(numerator, den, out=np.zeros_like(numerator), where=den > 1e-12)


def audit_clients(sketcher, payloads, sketches, clip_norm, slack=8.0):
    """Per-client consistency between the decoded payload and the sketch the client declared.

    The tolerance depends only on server-known constants (projection operator norm and clip norm),
    never on client-supplied quantities. Returns ``(excluded_reasons, residuals, tolerance)``."""
    eps = np.finfo(np.float32).eps
    tol = float(slack * eps * sketcher.opnorm * max(float(clip_norm), 1e-12) * math.sqrt(max(sketcher.ds, 1)))
    tol = max(tol, 1e-09)
    declared_cap = sketcher.opnorm * float(clip_norm) * (1 + 0.001)
    bad = {}
    residuals = {}
    for k in sorted(payloads):
        expected = sketcher(payloads[k])
        r = float(np.linalg.norm(expected - np.asarray(sketches[k], float)))
        residuals[k] = r
        if r > tol:
            bad[k] = "inconsistent_sketch"
        elif float(np.linalg.norm(sketches[k])) > declared_cap:
            bad[k] = "oversized_sketch"
    return (bad, residuals, tol)


ATTACK_SCALES = dict(sign_flip=5.0, scaled=5.0, gaussian=5.0, sketch_forgery=2.5, nullspace=5.0)


def attack_update(kind, update, rng, sketcher=None, scale=None):
    """Simulated malicious client behaviour, used to exercise screening and audit."""
    s = ATTACK_SCALES.get(kind, 5.0) if scale is None else float(scale)
    if kind in ("clean", "label_flip", "free_rider"):
        return np.zeros_like(update) if kind == "free_rider" else update.copy()
    if kind == "sign_flip":
        return -s * update
    if kind == "sketch_forgery":
        return -s * update
    if kind == "scaled":
        return s * update
    if kind == "gaussian":
        return rng.normal(size=len(update)) * s * max(np.linalg.norm(update), 1e-08) / np.sqrt(len(update))
    if kind == "nullspace":
        return update + s * max(np.linalg.norm(update), 1e-08) * sketcher.null_vector(rng)
    raise ValueError(kind)
