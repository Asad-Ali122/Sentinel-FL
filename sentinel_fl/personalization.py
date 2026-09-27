"""Gated, shrunk-bias on-device personalisation from a short labelled support prefix."""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.special import expit

from .metrics import metric_dict
from .preprocessing import assert_temporal_disjoint, support_query


def fit_personal_bias(z, y, ridge):
    """Ridge-penalised logistic bias fitted on a client's support logits."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    if not len(z):
        return 0.0

    def fun(b):
        return float(np.sum(np.logaddexp(0, z + b) - y * (z + b)) + 0.5 * ridge * b * b)

    res = optimize.minimize_scalar(fun, bounds=(-3.0, 3.0), method="bounded")
    if not res.success:
        raise RuntimeError("Personal bias fit failed")
    return float(res.x)


def loo_bias_is_helpful(z, y, ridge, margin=0.03):
    """Leave-one-out check that the shrunk bias beats no adaptation on unseen support points.

    The adapted loss must be lower by a relative ``margin``. Returns ``(helpful, base_nll, adapted_nll)``."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    n = len(z)
    if n < 8:
        return (False, np.nan, np.nan)
    base = 0.0
    adapted = 0.0
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        b = shrunk_personal_bias(z[m], y[m], ridge)
        base += float(np.logaddexp(0, z[i]) - y[i] * z[i])
        zi = z[i] + b
        adapted += float(np.logaddexp(0, zi) - y[i] * zi)
    base /= n
    adapted /= n
    return (bool(adapted < base * (1.0 - float(margin))), float(base), float(adapted))


def shrunk_personal_bias(z, y, ridge, shrink_k=15.0, cap=1.0):
    """Bias shrunk toward zero by ``n / (n + shrink_k)`` and clipped to ``+-cap``."""
    n = len(z)
    if n == 0:
        return 0.0
    b = fit_personal_bias(z, y, ridge) * (n / (n + float(shrink_k)))
    return float(np.clip(b, -abs(cap), abs(cap)))


def personalization_rows(predict_logits, calibrator, decoder, test, cfg, threshold, minority=1):
    """Evaluate the shared model and its personalised variant on a held-out subject's query suffix.

    The first ``cfg.adapt_shots`` epochs are the labelled support set; evaluation uses only the
    later, temporally purged query epochs. The personal bias is applied only if the support set
    contains enough minority examples, the logits are not constant, and a leave-one-out check on
    the support set shows that the bias helps. Returns one metrics row per arm (``shared`` and
    ``personalised``); the list is empty if the recording is too short.
    """
    rows = []
    support, query = support_query(test, cfg.adapt_shots)
    if len(support) < 5 or len(query) < 10:
        return rows
    assert_temporal_disjoint(support, query)
    zs = calibrator.a * np.clip(predict_logits(support), -30, 30) + calibrator.b
    ys = support.y.to_numpy(int)
    counts = np.bincount(ys, minlength=2)
    logit_sd = float(np.std(zs))
    both = bool(counts.min() >= max(1, int(cfg.adapt_min_minority)) and logit_sd >= float(cfg.adapt_min_logit_sd))
    qz = calibrator.a * np.clip(predict_logits(query), -30, 30) + calibrator.b
    q_plain = expit(qz)
    q_base = decoder.smooth(query, q_plain) if cfg.sequence_decode else q_plain
    eligible = both or not cfg.adapt_require_both_classes
    bias = shrunk_personal_bias(zs, ys, cfg.adapt_ridge, cap=cfg.adapt_max_bias) if eligible else 0.0
    if eligible:
        helpful, loo_base, loo_adapted = loo_bias_is_helpful(zs, ys, cfg.adapt_ridge, cfg.adapt_loo_margin)
    else:
        helpful, loo_base, loo_adapted = False, np.nan, np.nan
    gate = bool(eligible and (helpful or not cfg.adapt_gate_by_loo))
    q_bias = expit(qz + bias)
    if cfg.sequence_decode:
        q_bias = decoder.smooth(query, q_bias)
    arms = [("shared", q_base, 0.0), ("personalised", q_bias if gate else q_base, bias if gate else 0.0)]
    for arm, qprob, b_use in arms:
        row = metric_dict(query.y.to_numpy(int), qprob, threshold, minority)
        row.update(
            arm=arm,
            support_n=len(support),
            query_n=len(query),
            support_classes=int(support.y.nunique()),
            support_minority_n=int(counts.min()),
            adaptation_eligible=bool(eligible),
            support_logit_sd=logit_sd,
            personal_bias=float(b_use),
            adaptation_gate_passed=bool(gate),
            loo_nll_shared=loo_base,
            loo_nll_personalised=loo_adapted,
        )
        rows.append(row)
    return rows
