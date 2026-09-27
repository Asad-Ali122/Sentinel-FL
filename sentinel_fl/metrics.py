"""Evaluation metrics, probability calibration, threshold selection and bootstrap intervals."""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .preprocessing import subject_weights
from .utils import rng_for


def metric_dict(y, p, threshold=0.5, minority=1):
    """Threshold-dependent and threshold-free metrics of one evaluation set.

    A participant with a single class is flagged by ``degenerate_subject``."""
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-07, 1 - 1e-07)
    if len(y) != len(p) or not len(y) or (not np.isfinite(p).all()):
        raise ValueError("Invalid predictions")
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n0 = tn + fp
    n1 = tp + fn

    def divide(a, b):
        return float(a / b) if b else np.nan

    sensitivity = divide(tp, n1)
    specificity = divide(tn, n0)
    auc = float(roc_auc_score(y, p)) if n0 and n1 else np.nan
    ap1 = float(average_precision_score(y, p)) if n1 and n0 else np.nan
    ap0 = float(average_precision_score(1 - y, 1 - p)) if n1 and n0 else np.nan
    ece = 0.0
    cece = 0.0
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == y
    for b in range(10):
        m = np.minimum((p * 10).astype(int), 9) == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
        m = np.minimum((conf * 10).astype(int), 9) == b
        if m.any():
            cece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    ap = ap1 if minority else ap0
    base = float(np.mean(y == minority))
    return dict(
        n=len(y),
        n_class0=int(n0),
        n_class1=int(n1),
        prevalence=float(y.mean()),
        degenerate_subject=bool(n0 == 0 or n1 == 0),
        macro_f1=float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        f1_class0=float(f1_score(y, pred, labels=[0], average="macro", zero_division=0)),
        f1_class1=float(f1_score(y, pred, labels=[1], average="macro", zero_division=0)),
        accuracy=float(np.mean(pred == y)),
        balanced_acc=(sensitivity + specificity) / 2,
        sensitivity=sensitivity,
        specificity=specificity,
        auroc=auc,
        ap_class0=ap0,
        ap_class1=ap1,
        minority_ap=ap,
        minority_ap_lift=divide(ap, base),
        probability_ece=float(ece),
        confidence_ece=float(cece),
        brier=float(np.mean((y - p) ** 2)),
        nll=float(log_loss(y, p, labels=[0, 1])),
        kappa=float(cohen_kappa_score(y, pred, labels=[0, 1])) if n0 and n1 else np.nan,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        threshold=float(threshold),
    )


def participant_score(df, p, threshold=0.5, metric="macro_f1"):
    """Mean of a metric across participants, each scored on its own rows."""
    scores = []
    for cid in sorted(df.client_id.unique()):
        m = df.client_id.to_numpy() == cid
        scores.append(metric_dict(df.y.to_numpy()[m], np.asarray(p)[m], threshold)[metric])
    return float(np.nanmean(scores))


def select_threshold(df, raw_p, grid=None):
    """Threshold maximising the participant-mean macro-F1 (ties broken toward 0.5)."""
    grid = np.linspace(0.02, 0.98, 193) if grid is None else np.asarray(grid, float)
    vals = [participant_score(df, raw_p, t) for t in grid]
    best = max(vals)
    ix = [i for i, v in enumerate(vals) if abs(v - best) < 1e-12]
    return float(grid[min(ix, key=lambda i: abs(grid[i] - 0.5))])


class Calibrator:
    """Two-parameter (scale, bias) logistic calibration fitted with participant-balanced weights."""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.status = "identity"

    def fit(self, logits, df):
        z = np.clip(np.asarray(logits, float), -30, 30)
        y = df.y.to_numpy(float)
        w = subject_weights(df)
        if len(np.unique(y)) < 2:
            self.status = "identity_single_class"
            return self

        def objective(v):
            a = np.exp(v[0])
            zz = a * z + v[1]
            loss = np.average(np.logaddexp(0, zz) - y * zz, weights=w) + 0.005 * (v @ v)
            return float(loss)

        res = optimize.minimize(objective, [0.0, 0.0], method="L-BFGS-B", bounds=[(-2.3, 2.3), (-5, 5)])
        if res.success and np.isfinite(res.fun):
            self.a = float(np.exp(res.x[0]))
            self.b = float(res.x[1])
            self.status = "fitted"
        else:
            self.status = "identity_optimizer_failed"
        return self

    def predict(self, z):
        return expit(self.a * np.clip(z, -30, 30) + self.b)

    def threshold(self, raw_t):
        return float(self.predict(np.log(raw_t / (1 - raw_t))))


def bootstrap_mean(x, reps, seed, min_points=6):
    """Percentile bootstrap 95% interval of the mean; ``(nan, nan)`` below ``min_points`` values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < max(2, int(min_points)):
        return (np.nan, np.nan)
    rng = rng_for("bootstrap", seed)
    vals = []
    for start in range(0, reps, 250):
        ix = rng.integers(0, len(x), (min(250, reps - start), len(x)))
        vals.extend(x[ix].mean(axis=1))
    return tuple(np.quantile(vals, [0.025, 0.975]))
