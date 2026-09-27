"""Leave-one-subject-out splits, leakage audits and fit-subject-only preprocessing."""

from __future__ import annotations

import numpy as np

from .config import SENSORS
from .utils import rng_for


def make_splits(df, cfg):
    """Leave-one-subject-out splits with disjoint fit / validation / calibration subject sets.

    For every held-out subject the remaining subjects are partitioned by a seeded permutation
    (never by outcomes) into ``fit``, ``val`` (threshold, decoder strength, learning rate) and
    ``cal`` (probability calibration and transition matrix). ``cfg.max_folds`` selects a fixed
    random subset of held-out subjects."""
    splits = []
    for dataset, g in df.groupby("dataset", sort=True):
        ids = sorted(g.client_id.unique())
        if len(ids) < 5:
            raise ValueError(f"{dataset}: need at least five subjects")
        helds = ids
        if cfg.max_folds is not None:
            ix = rng_for("pilot-folds", dataset).permutation(len(ids))[: cfg.max_folds]
            helds = [ids[i] for i in sorted(ix)]
        for held in helds:
            rest = [s for s in ids if s != held]
            order = rng_for("inner-partitions", dataset, held).permutation(rest).tolist()
            nv = min(cfg.val_subjects, max(1, (len(rest) - 3) // 2))
            val, cal = (order[:nv], order[nv : 2 * nv])
            fit = order[2 * nv :]
            rec = dict(dataset=dataset, held=held, fit=fit, val=val, cal=cal, test=[held])
            audit_partitions(rec)
            splits.append(rec)
    return splits


def audit_partitions(split):
    """Raise if any data role is empty or two roles share a participant."""
    sets = [set(split[k]) for k in ("fit", "val", "cal", "test")]
    if any(not s for s in sets):
        raise AssertionError("Empty data role")
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            if a & b:
                raise AssertionError("Participant leakage: " + str(a & b))
    return True


def role_frame(df, sp, role):
    """Rows belonging to the participants assigned to ``role``."""
    return df[df.client_id.isin(sp[role])].copy()


def support_query(g, shots):
    """Split one subject's recording into a leading support set and a purged query suffix.

    The query starts only after the latest feature/label extent of the support set."""
    g = g.sort_values("label_start")
    n = min(int(shots), len(g) // 3)
    if n < 5:
        return (g.iloc[:0], g.iloc[:0])
    support = g.iloc[:n]
    cutoff = float(np.maximum(support.feature_end, support.label_end).max())
    query = g[(g.feature_start >= cutoff) & (g.label_start >= cutoff)]
    return (support, query)


def assert_temporal_disjoint(a, b):
    """Raise if two frames share rows or overlap in time within a participant."""
    if set(a.index) & set(b.index):
        raise AssertionError("Shared rows")
    for cid in set(a.client_id) & set(b.client_id):
        x = a[a.client_id == cid]
        y = b[b.client_id == cid]
        lo = np.minimum(x.feature_start, x.label_start).to_numpy()
        hi = np.maximum(x.feature_end, x.label_end).to_numpy()
        for _, r in y.iterrows():
            s = min(r.feature_start, r.label_start)
            e = max(r.feature_end, r.label_end)
            if np.any((lo < e) & (hi > s)):
                raise AssertionError("Overlapping temporal support")
    return True


class Preprocessor:
    """Robust per-feature standardisation fitted on fit-subject rows only.

    Missing values are imputed as zero after scaling and flagged by one missingness indicator per
    sensor group, so the model input has ``len(features) + len(groups)`` dimensions."""

    def fit(self, df, clip=8.0):
        self.fit_ids = tuple(sorted(df.client_id.unique()))
        self.clip = float(clip)
        cand = sorted(c for c in df if c.split("_")[0] in SENSORS)
        keep = []
        med = []
        scale = []
        for c in cand:
            a = df[c].to_numpy(float)
            v = a[np.isfinite(a)]
            if len(v) < 2 or np.std(v) < 1e-10:
                continue
            q = np.percentile(v, [25, 50, 75])
            s = (q[2] - q[0]) / 1.349
            keep.append(c)
            med.append(q[1])
            scale.append(max(s, np.std(v) * 0.05, 1e-06))
        if not keep:
            raise ValueError("No variable fit-subject features")
        self.features = keep
        self.med = np.array(med)
        self.scale = np.array(scale)
        self.groups = {
            g: np.array([i for i, c in enumerate(keep) if c.startswith(g + "_")], int)
            for g in SENSORS
            if any(c.startswith(g + "_") for c in keep)
        }
        self.dimension = len(keep) + len(self.groups)
        return self

    def transform(self, df):
        a = df.reindex(columns=self.features).to_numpy(float)
        seen = np.isfinite(a)
        x = np.where(seen, np.clip((a - self.med) / self.scale, -self.clip, self.clip), 0.0)
        missing = np.column_stack([1 - seen[:, j].mean(axis=1) for j in self.groups.values()])
        return np.column_stack([x, missing])

    def available_inputs(self, df):
        a = df.reindex(columns=self.features).to_numpy(float)
        return np.r_[np.isfinite(a).any(axis=0), np.ones(len(self.groups), bool)]


def subject_weights(df):
    """Row weights that give every participant equal total mass (mean-normalised)."""
    counts = df.client_id.map(df.client_id.value_counts()).to_numpy(float)
    w = 1 / counts
    return w / w.mean()


def class_weights(df, power=1.0):
    """Participant-balanced, inverse-frequency class weights (mean-normalised over the fit population)."""
    w = subject_weights(df)
    cnt = np.bincount(df.y.astype(int), weights=w, minlength=2)
    if (cnt == 0).any():
        raise ValueError("Fit population must contain both labels")
    cw = (cnt.sum() / (2 * cnt)) ** power
    return cw / (cw @ cnt / cnt.sum())
