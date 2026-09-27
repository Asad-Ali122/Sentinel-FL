"""Two-state sequence decoder applied to each subject's own probability stream."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .features import contiguous_runs
from .metrics import participant_score, select_threshold


class SequenceDecoder:
    """Two-state hidden-Markov smoother over each subject's own probability sequence.

    The transition matrix is estimated from calibration subjects' label sequences; the smoothed
    log-odds are blended back with a strength (0 = no smoothing) chosen on validation subjects."""

    def __init__(self, strengths=(0.0, 0.25, 0.5, 0.75, 1.0), min_gap_ratio=1.75):
        self.strengths = tuple(float(s) for s in strengths)
        self.min_gap_ratio = float(min_gap_ratio)
        self.A = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.pi = np.array([0.5, 0.5])
        self.strength = 0.0
        self.status = "identity"

    @staticmethod
    def _subject_runs(times, gap_ratio):
        times = np.asarray(times, float)
        if len(times) < 2:
            return [np.arange(len(times))]
        step = float(np.median(np.diff(times)))
        step = step if np.isfinite(step) and step > 0 else 1.0
        return contiguous_runs(times, step, gap_ratio)

    def fit(self, df):
        counts = np.ones((2, 2)) * 0.5
        prior = np.ones(2) * 0.5
        for cid, g in df.groupby("client_id", sort=True):
            g = g.sort_values("label_start")
            y = g.y.to_numpy(int)
            t = g.label_start.to_numpy(float)
            prior += np.bincount(y, minlength=2)
            for run in self._subject_runs(t, self.min_gap_ratio):
                if len(run) < 2:
                    continue
                yy = y[run]
                for a, b in zip(yy[:-1], yy[1:]):
                    counts[a, b] += 1
        self.A = counts / counts.sum(axis=1, keepdims=True)
        self.pi = prior / prior.sum()
        self.status = "fitted"
        return self

    def _smooth_one(self, p, times):
        """Forward-backward smoothing of one contiguous run of posteriors."""
        p = np.clip(np.asarray(p, float), 1e-06, 1 - 1e-06)
        out = np.array(p, copy=True)
        logA = np.log(np.clip(self.A, 1e-12, None))
        logpi = np.log(np.clip(self.pi, 1e-12, None))
        for run in self._subject_runs(times, self.min_gap_ratio):
            if len(run) == 0:
                continue
            q = p[run]
            T = len(q)
            loge = np.column_stack([np.log(1 - q), np.log(q)]) - logpi[None, :]
            al = np.empty((T, 2))
            be = np.empty((T, 2))
            al[0] = logpi + loge[0]
            for t in range(1, T):
                al[t] = loge[t] + logsumexp(al[t - 1][:, None] + logA, axis=0)
            be[T - 1] = 0.0
            for t in range(T - 2, -1, -1):
                be[t] = logsumexp(logA + (loge[t + 1] + be[t + 1])[None, :], axis=1)
            g = al + be
            g -= logsumexp(g, axis=1, keepdims=True)
            out[run] = np.clip(np.exp(g[:, 1]), 1e-06, 1 - 1e-06)
        return out

    def smooth(self, df, p, strength=None):
        s = self.strength if strength is None else float(strength)
        p = np.clip(np.asarray(p, float), 1e-06, 1 - 1e-06)
        if s <= 0.0 or self.status != "fitted":
            return p
        out = np.array(p, copy=True)
        ids = df.client_id.to_numpy()
        order = df.label_start.to_numpy(float)
        for cid in pd.unique(ids):
            m = np.flatnonzero(ids == cid)
            m = m[np.argsort(order[m], kind="stable")]
            out[m] = self._smooth_one(p[m], order[m])
        lo = np.log(p / (1 - p))
        ls = np.log(out / (1 - out))
        return expit((1 - s) * lo + s * ls)

    def tune(self, df_val, p_val):
        """Choose the blend strength and threshold on validation subjects only."""
        best = (-np.inf, 0.0, 0.5)
        for s in self.strengths:
            q = self.smooth(df_val, p_val, s)
            t = select_threshold(df_val, q)
            score = participant_score(df_val, q, t)
            if score > best[0] + 1e-12:
                best = (score, s, t)
        self.strength = float(best[1])
        return dict(
            strength=self.strength,
            val_macro_f1=float(best[0]),
            threshold=float(best[2]),
            status=self.status,
            stay_class0=float(self.A[0, 0]),
            stay_class1=float(self.A[1, 1]),
        )
