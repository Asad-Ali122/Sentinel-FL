"""Feature extraction for WESAD (stress) and the PSG-labelled sleep cohort, plus temporal-context features."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .config import SENSORS, VERSION, code_fingerprint
from .utils import atomic_pickle, digest, hash_frame, rng_for


def basic_stats(x, prefix, dt=1.0):
    """Summary statistics (moments, quantiles, slope, mean absolute difference) of a 1-D signal segment."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    q = np.percentile(x, [10, 25, 50, 75, 90])
    d = np.diff(x)
    t = np.arange(len(x)) * dt
    slope = float(np.dot(t - t.mean(), x - x.mean()) / max(np.dot(t - t.mean(), t - t.mean()), 1e-12))
    vals = dict(
        mean=x.mean(),
        std=x.std(),
        median=q[2],
        iqr=q[3] - q[1],
        p10=q[0],
        p90=q[4],
        mad=np.median(np.abs(x - q[2])),
        rms=np.sqrt(np.mean(x * x)),
        slope=slope,
        dmean=np.mean(np.abs(d)) if len(d) else 0.0,
        n=len(x),
    )
    return {f"{prefix}_{k}": float(v) for k, v in vals.items()}


def spectral_stats(x, fs, prefix, bands):
    """Welch spectral entropy, dominant frequency and relative band powers of a signal segment."""
    x = np.asarray(x, float).ravel()
    if len(x) < max(16, int(fs * 2)) or not np.isfinite(x).all():
        return {}
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), max(32, int(fs * 8))))
    total = float(np.sum(p))
    out = {}
    if total <= 1e-20:
        return {f"{prefix}_spec_entropy": 0.0}
    prob = p / total
    out[f"{prefix}_spec_entropy"] = float(-np.sum(prob * np.log(prob + 1e-15)) / np.log(len(prob)))
    out[f"{prefix}_dominant_hz"] = float(f[1 + np.argmax(p[1:])])
    for lo, hi in bands:
        out[f"{prefix}_power_{lo:g}_{hi:g}"] = float(p[(f >= lo) & (f < hi)].sum() / total)
    return out


def signal_features(x, fs, prefix):
    """Features of one physiological channel; the channel type is read from the prefix.

    ECG / PPG channels additionally yield beat-interval statistics; EDA and EMG yield band powers."""
    x = np.asarray(x, float).ravel()
    out = basic_stats(x, prefix, 1 / fs)
    if len(x) < 16 or not np.isfinite(x).all():
        return out
    kind = prefix.split("_")[0]
    if kind in ("ecg", "ppg", "resp"):
        band = {"ecg": (0.5, 35.0), "ppg": (0.5, 8.0), "resp": (0.05, 1.0)}[kind]
        lo, hi = band
        hi = min(hi, fs * 0.45)
        if lo >= hi:
            return out
        sos = signal.butter(3, [lo, hi], fs=fs, btype="bandpass", output="sos")
        xf = signal.sosfilt(sos, x)
        out.update(spectral_stats(xf, fs, prefix, [(lo, hi)]))
        if kind in ("ecg", "ppg"):
            if kind == "ecg":
                xf = xf if np.percentile(xf, 99) > abs(np.percentile(xf, 1)) else -xf
            peaks, _ = signal.find_peaks(xf, distance=max(1, int(0.3 * fs)), prominence=max(0.4 * xf.std(), 1e-12))
            intervals = np.diff(peaks) / fs
            good = intervals[(intervals >= 0.3) & (intervals <= 2.0)]
            out[f"{prefix}_interval_valid_fraction"] = float(len(good) / max(len(intervals), 1))
            if len(good) >= 5 and len(good) >= 0.8 * len(intervals):
                out[f"{prefix}_pulse_rate_median"] = float(60 / np.median(good))
                out[f"{prefix}_interval_sd"] = float(good.std())
                valid = (intervals >= 0.3) & (intervals <= 2.0)
                dd = np.diff(intervals)[valid[1:] & valid[:-1]]
                if len(dd) >= 3:
                    out[f"{prefix}_interval_rmssd"] = float(np.sqrt(np.mean(dd * dd)))
    elif kind == "eda":
        out.update(spectral_stats(x, fs, prefix, [(0.0, 0.05), (0.05, 0.5)]))
        out[f"{prefix}_positive_diff_mean"] = float(np.maximum(np.diff(x), 0).mean())
    elif kind == "emg":
        out.update(spectral_stats(x, fs, prefix, [(20.0, 50.0), (50.0, 150.0), (150.0, 250.0)]))
    return out


def accel_features(a, prefix, fs=1.0):
    """Per-axis statistics, magnitude statistics and an activity index for an (n, 3) accelerometer segment."""
    a = np.asarray(a, float)
    if a.ndim != 2 or a.shape[1] < 3 or (not len(a)):
        return {}
    out = {}
    for j in range(3):
        out.update(basic_stats(a[:, j], f"{prefix}_a{j}", 1 / fs))
    mag = np.linalg.norm(a[:, :3], axis=1)
    out.update(basic_stats(mag, f"{prefix}_mag", 1 / fs))
    out[f"{prefix}_activity"] = float(np.mean(np.abs(mag - np.median(mag))))
    return out


def label_gate(labels, valid, coverage=0.95, purity=0.95):
    """Accept a window only if enough of it carries a valid label and one label dominates.

    Returns ``(dominant_label_or_None, coverage, purity)``."""
    a = np.asarray(labels, int)
    ok = np.isin(a, list(valid))
    cov = float(ok.mean()) if len(a) else 0.0
    if not ok.any():
        return (None, cov, 0.0)
    vals, counts = np.unique(a[ok], return_counts=True)
    dom = int(vals[np.argmax(counts)])
    pur = float(counts.max() / len(a))
    return (dom if cov >= coverage and pur >= purity else None, cov, pur)


def load_wesad(path, cfg):
    """Extract sliding-window features and labels from one WESAD subject pickle.

    Only load trusted local WESAD files: ``pickle`` can execute arbitrary code.
    Returns ``(frame, log_entry)``."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    labels = np.asarray(d["label"]).ravel().astype(int)
    sid = Path(path).stem
    W = cfg["wesad_window_s"]
    stride = cfg["wesad_stride_s"]
    valid = {1, 2, 3} if cfg["include_amusement_as_nonstress"] else {1, 2}
    rows = []
    excluded = 0
    for t in np.arange(0, len(labels) / 700 - W + 1e-07, stride):
        lab = labels[int(round(t * 700)) : int(round((t + W) * 700))]
        y, cov, pur = label_gate(lab, valid, cfg["min_valid_coverage"], cfg["min_label_purity"])
        if y is None:
            excluded += 1
            continue
        feats = {}
        mapping = [
            ("wrist", "ACC", 32, "acc_wrist"),
            ("wrist", "BVP", 64, "ppg_wrist"),
            ("wrist", "EDA", 4, "eda_wrist"),
            ("wrist", "TEMP", 4, "temp_wrist"),
        ]
        if cfg["sensor_profile"] == "all":
            mapping += [
                ("chest", k, 700, p)
                for k, p in [
                    ("ACC", "acc_chest"),
                    ("ECG", "ecg_chest"),
                    ("EDA", "eda_chest"),
                    ("EMG", "emg_chest"),
                    ("Resp", "resp_chest"),
                    ("Temp", "temp_chest"),
                ]
            ]
        for device, key, fs, prefix in mapping:
            a = d["signal"].get(device, {}).get(key)
            if a is None and key == "Temp":
                a = d["signal"].get(device, {}).get("TEMP")
            if a is None:
                continue
            seg = np.asarray(a)[int(round(t * fs)) : int(round((t + W) * fs))]
            if len(seg) < 0.9 * W * fs:
                continue
            feats.update(accel_features(seg, prefix, fs) if key == "ACC" else signal_features(seg, fs, prefix))
        if feats:
            rows.append(
                dict(
                    dataset="WESAD",
                    client_id=f"WESAD_{sid}",
                    task="stress",
                    y=int(y == 2),
                    feature_start=t,
                    feature_end=t + W,
                    label_start=t,
                    label_end=t + W,
                    label_coverage=cov,
                    label_purity=pur,
                    **feats,
                )
            )
    return (pd.DataFrame(rows), dict(dataset="WESAD", subject=sid, kept=len(rows), excluded=excluded))


def read_numeric(path, ncols):
    """Read a whitespace/comma separated numeric file, sorted by its first (time) column."""
    p = Path(path)
    if not p.exists():
        return np.empty((0, ncols))
    a = pd.read_csv(p, sep="\\s+|,", engine="python", header=None, comment="#")
    a = a.apply(pd.to_numeric, errors="coerce").dropna(subset=[0]).to_numpy(float)
    if a.shape[1] < ncols:
        raise ValueError(f"Bad numeric file: {p}")
    a = a[:, :ncols]
    return a[np.argsort(a[:, 0], kind="stable")]


def segment(a, start, end):
    """Rows of a time-sorted array whose first column lies in ``[start, end)``."""
    if not len(a):
        return a
    lo, hi = np.searchsorted(a[:, 0], [start, end], side="left")
    return a[lo:hi]


def complete_steps(a, start, end):
    """Step counts whose bin has fully ended inside ``[start, end]``.

    A count stamped at time ``t_i`` covers ``[t_i, t_{i+1})`` and is only known at ``t_{i+1}``;
    the last bin has an unknown end and is therefore excluded."""
    if len(a) < 2:
        return np.empty(0)
    ok = (a[:-1, 0] >= start) & (a[1:, 0] <= end) & (a[1:, 0] > a[:-1, 0])
    return a[:-1, 1][ok]


def load_sleep(sid, cfg):
    """Extract 30-second epoch features and wake/sleep labels for one sleep-study subject."""
    hr = read_numeric(Path(cfg["sleep_heart_dir"]) / f"{sid}_heartrate.txt", 2)
    acc = read_numeric(Path(cfg["sleep_motion_dir"]) / f"{sid}_acceleration.txt", 4)
    lab = read_numeric(Path(cfg["sleep_label_dir"]) / f"{sid}_labeled_sleep.txt", 2)
    steps = read_numeric(Path(cfg["sleep_steps_dir"]) / f"{sid}_steps.txt", 2)
    E = cfg["sleep_epoch_s"]
    rows = []
    excluded = 0
    if cfg["sleep_label_position"] not in ("start", "end"):
        raise ValueError("label position")
    for ts, stage in lab:
        if not np.isfinite(stage) or stage not in (0, 1, 2, 3, 5):
            excluded += 1
            continue
        end = ts + E if cfg["sleep_label_position"] == "start" else ts
        begin = end - E
        feats = {}
        support = begin
        am = segment(acc, begin, end)
        if len(am) < 2 or am[-1, 0] - am[0, 0] < 0.8 * E:
            excluded += 1
            continue
        fs_est = 1 / max(np.median(np.diff(am[:, 0])), 1e-06)
        feats.update(accel_features(am[:, 1:4], "acc_watch", fs_est))
        hm = segment(hr, begin, end)
        hm = hm[(hm[:, 1] > 0) & np.isfinite(hm[:, 1])]
        if len(hm):
            feats.update(basic_stats(hm[:, 1], "hr_epoch"))
        ctx = end - cfg["sleep_context_s"]
        amc = segment(acc, ctx, end)
        hmc = segment(hr, ctx, end)
        hmc = hmc[(hmc[:, 1] > 0) & np.isfinite(hmc[:, 1])]
        if len(amc):
            feats.update(basic_stats(np.linalg.norm(amc[:, 1:4], axis=1), "acc_context_mag"))
            support = min(support, ctx)
        if len(hmc):
            feats.update(basic_stats(hmc[:, 1], "hr_context"))
            support = min(support, ctx)
        st = end - cfg["steps_context_s"]
        sv = complete_steps(steps, st, end)
        if len(sv):
            feats.update(basic_stats(sv, "steps_complete"))
            support = min(support, st)
        rows.append(
            dict(
                dataset="SLEEP",
                client_id=f"SLEEP_{sid}",
                task="sleep",
                y=int(stage != 0),
                feature_start=support,
                feature_end=end,
                label_start=begin,
                label_end=end,
                label_coverage=1.0,
                label_purity=1.0,
                **feats,
            )
        )
    return (pd.DataFrame(rows), dict(dataset="SLEEP", subject=sid, kept=len(rows), excluded=excluded))


def contiguous_runs(times, step, gap_ratio=1.75):
    """Split epoch indices wherever the recording has a gap larger than ``gap_ratio`` epochs."""
    times = np.asarray(times, float)
    if len(times) < 2:
        return [np.arange(len(times))]
    brk = np.flatnonzero(np.diff(times) > gap_ratio * step) + 1
    return np.split(np.arange(len(times)), brk)


def _roll(series, window, centered, reverse=False, stat="mean"):
    s = series[::-1] if reverse else series
    r = s.rolling(int(window), center=bool(centered), min_periods=1)
    out = r.mean() if stat == "mean" else r.std(ddof=0)
    out = out[::-1] if reverse else out
    return out.to_numpy(float)


def robust_scale(v):
    """Robust standard-deviation estimate (IQR/1.349, floored) of a vector."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return 1.0
    q = np.percentile(v, [25, 75])
    return float(max((q[1] - q[0]) / 1.349, np.std(v) * 0.05, 1e-06))


def add_context_features(df, cfg):
    """Append per-subject temporal-context, subject-relative and elapsed-time features.

    Every statistic is computed strictly inside one subject's own recording, so the transform is
    valid for a held-out subject and computable on the device. Rolling windows may look forward in
    time only where the dataset configuration allows it; the resulting look-ahead is declared per
    row in ``declared_lookahead_s`` and checked by :func:`validate_frame`."""
    if not cfg.get("context_enabled", True):
        return (df, dict(enabled=False))
    centered = bool(cfg.get("context_centered", True))
    subject_rel = bool(cfg.get("context_subject_relative", True))
    time_feats = bool(cfg.get("context_time_features", True))
    base_map = cfg.get("context_base_columns", {})
    scale_map = cfg.get("context_scales_by_dataset", {})
    fwd_map = cfg.get("context_forward_by_dataset", {})
    default_scales = tuple(cfg.get("context_scales", (4, 12, 40)))
    report = dict(
        enabled=True, centered=centered, subject_relative=subject_rel, time_features=time_feats, per_dataset={}
    )
    pieces = []
    for dataset, gd in df.groupby("dataset", sort=True):
        base = [c for c in base_map.get(dataset, ()) if c in gd.columns]
        base = [c for c in base if np.isfinite(gd[c].to_numpy(float)).any()]
        scales = tuple(scale_map.get(dataset, default_scales))
        use_forward = bool(fwd_map.get(dataset, centered))
        report["per_dataset"][dataset] = dict(
            base_columns=base, scales=list(scales), forward=use_forward, n_base=len(base)
        )
        if not base:
            pieces.append(gd)
            continue
        out_parts = []
        for cid, g in gd.groupby("client_id", sort=True):
            g = g.sort_values("label_start").copy()
            t = g.label_start.to_numpy(float)
            step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
            step = step if np.isfinite(step) and step > 0 else 1.0
            runs = contiguous_runs(t, step)
            new = {}
            for c in base:
                x = g[c].to_numpy(float)
                sx = pd.Series(x).interpolate(limit_direction="both")
                filled = sx.to_numpy(float)
                if not np.isfinite(filled).any():
                    continue
                filled = np.nan_to_num(
                    filled, nan=float(np.nanmedian(filled)) if np.isfinite(np.nanmedian(filled)) else 0.0
                )
                sfill = pd.Series(filled)
                scl = robust_scale(filled)
                med = float(np.median(filled))
                for w in scales:
                    cm = np.full(len(g), np.nan)
                    cs = np.full(len(g), np.nan)
                    bm = np.full(len(g), np.nan)
                    fm = np.full(len(g), np.nan)
                    for run in runs:
                        if not len(run):
                            continue
                        seg = sfill.iloc[run]
                        cm[run] = _roll(seg, w, centered, stat="mean")
                        cs[run] = _roll(seg, w, centered, stat="std")
                        bm[run] = _roll(seg, w, False, stat="mean")
                        if use_forward:
                            fm[run] = _roll(seg, w, False, reverse=True, stat="mean")
                    new[f"ctx_{c}_cm{w}"] = (cm - med) / scl
                    new[f"ctx_{c}_cs{w}"] = cs / scl
                    new[f"ctx_{c}_dev{w}"] = (filled - cm) / scl
                    new[f"ctx_{c}_bm{w}"] = (bm - med) / scl
                    if use_forward:
                        new[f"ctx_{c}_fm{w}"] = (fm - med) / scl
                        new[f"ctx_{c}_asym{w}"] = (fm - bm) / scl
                if subject_rel:
                    new[f"ctx_{c}_z"] = (filled - med) / scl
                    rank = pd.Series(filled).rank(pct=True).to_numpy(float)
                    new[f"ctx_{c}_pct"] = rank - 0.5
            if time_feats and len(t):
                t0 = float(t.min())
                t1 = float(t.max())
                span = max(t1 - t0, 1e-06)
                elapsed = (t - t0) / 60.0
                retro = str(cfg.get("context_time_mode", "retrospective")) == "retrospective"
                new["time_elapsed_min"] = elapsed
                new["time_log_elapsed"] = np.log1p(np.maximum(elapsed, 0.0))
                act_col = next((c for c in base if "activity" in c or "mag_mean" in c), None)
                a = None
                if act_col is not None:
                    a = pd.Series(g[act_col].to_numpy(float)).interpolate(limit_direction="both").to_numpy(float)
                    a = np.nan_to_num(a, nan=0.0)
                    a = np.maximum(a - np.median(a), 0.0)
                    new["time_mean_activity_so_far"] = np.cumsum(a) / np.arange(1, len(a) + 1)
                    thr = float(np.percentile(a, 75)) if len(a) > 3 else np.inf
                    since = np.empty(len(a))
                    last = -np.inf
                    for i, val in enumerate(a):
                        if val >= thr and np.isfinite(thr):
                            last = t[i]
                        since[i] = (t[i] - last) / 60.0 if np.isfinite(last) else 999.0
                    new["time_since_move_min"] = np.minimum(since, 999.0)
                if retro:
                    new["time_r_elapsed_frac"] = (t - t0) / span
                    new["time_r_remaining_min"] = (t1 - t) / 60.0
                    new["time_r_progress_cos"] = np.cos(2 * np.pi * (t - t0) / span)
                    new["time_r_progress_sin"] = np.sin(2 * np.pi * (t - t0) / span)
                    if a is not None:
                        tot = float(a.sum())
                        new["time_r_cum_activity_frac"] = np.cumsum(a) / tot if tot > 0 else np.zeros(len(a))
            if new:
                add = pd.DataFrame(new, index=g.index)
                g = pd.concat([g, add], axis=1)
                back = max(scales) * step
                fwd = max(scales) * step / 2.0 if centered or use_forward else 0.0
                g["feature_start"] = np.minimum(g.feature_start.to_numpy(float), t - back)
                g["feature_end"] = np.maximum(g.feature_end.to_numpy(float), g.label_end.to_numpy(float) + fwd)
                g["declared_lookahead_s"] = float(fwd)
            out_parts.append(g)
        pieces.append(pd.concat(out_parts, ignore_index=False))
    merged = pd.concat(pieces, ignore_index=False)
    if "declared_lookahead_s" not in merged:
        merged["declared_lookahead_s"] = 0.0
    merged["declared_lookahead_s"] = merged["declared_lookahead_s"].fillna(0.0)
    report["max_declared_lookahead_s"] = float(merged["declared_lookahead_s"].max())
    report["n_context_features"] = int(sum(1 for c in merged.columns if c.startswith(("ctx_", "time_"))))
    return (merged, report)


def validate_frame(df, max_lookahead_s=0.0):
    """Validate schema, labels, intervals, duplicates and the declared look-ahead of a feature frame.

    Returns the frame sorted by ``(dataset, client_id, label_start)``."""
    need = {"dataset", "client_id", "task", "y", "feature_start", "feature_end", "label_start", "label_end"}
    if not need.issubset(df):
        raise ValueError(f"Missing columns: {need - set(df)}")
    if not len(df) or df.client_id.isna().any():
        raise ValueError("Empty/bad cohort")
    if not df.y.isin([0, 1]).all():
        raise ValueError("Labels must be binary")
    if not (df.feature_start < df.feature_end).all():
        raise ValueError("Invalid feature interval")
    if not (df.label_start < df.label_end).all():
        raise ValueError("Invalid label interval")
    declared = df["declared_lookahead_s"].to_numpy(float) if "declared_lookahead_s" in df else np.zeros(len(df))
    if not np.isfinite(declared).all() or (declared < 0).any():
        raise ValueError("Bad declared lookahead")
    if declared.max() > float(max_lookahead_s) + 1e-08:
        raise ValueError(f"Declared lookahead {declared.max():g}s exceeds the configured bound {max_lookahead_s:g}s")
    overshoot = df.feature_end.to_numpy(float) - df.label_end.to_numpy(float)
    if not np.all(overshoot <= declared + 1e-08):
        raise ValueError("Features reach beyond prediction time by more than the declared lookahead")
    if df.duplicated(["client_id", "label_start"]).any():
        raise ValueError("Duplicate epochs")
    if (df.groupby("client_id").dataset.nunique() > 1).any():
        raise ValueError("Ambiguous client ID")
    features = [c for c in df if c.split("_")[0] in SENSORS]
    if not features:
        raise ValueError("No sensor features")
    if np.isinf(df[features].to_numpy(float)).any():
        raise ValueError("Infinite features")
    return df.sort_values(["dataset", "client_id", "label_start"]).reset_index(drop=True)


def synthetic_fixture(n_subjects=8, n_windows=100, seed=101, cfg=None):
    """Small synthetic two-state cohort for software tests only; never used for real experiments."""
    rng = rng_for("fixture", seed)
    rows = []
    for dataset, task in [("WESAD", "stress"), ("SLEEP", "sleep")]:
        for i in range(n_subjects):
            shift = rng.normal(0, 0.35, 8)
            state = int(rng.integers(0, 2))
            for t in range(n_windows):
                if rng.random() < 0.06:
                    state = 1 - state
                x = (2 * state - 1) * np.array([1, 0.5, -0.8, 0.6, 1.0, -0.3, 0.4, 0.2]) + shift + rng.normal(0, 0.8, 8)
                feat = {
                    f"{g}_f{k}": x[k] for k, g in enumerate(["acc", "acc", "hr", "hr", "eda", "eda", "temp", "temp"])
                }
                if dataset == "SLEEP":
                    feat.update(eda_f4=np.nan, eda_f5=np.nan, temp_f6=np.nan, temp_f7=np.nan)
                rows.append(
                    dict(
                        dataset=dataset,
                        task=task,
                        client_id=f"{dataset}_FIX{i}",
                        y=state,
                        feature_start=t * 30 - 60,
                        feature_end=t * 30 + 30,
                        label_start=t * 30,
                        label_end=t * 30 + 30,
                        label_coverage=1.0,
                        label_purity=1.0,
                        **feat,
                    )
                )
    df = pd.DataFrame(rows)
    cfg = cfg or {}
    fixture_cfg = dict(cfg)
    fixture_cfg.setdefault("context_enabled", False)
    df, _ = add_context_features(df, fixture_cfg)
    return validate_frame(df, fixture_cfg.get("context_max_lookahead_s", 0.0))


def load_data(cfg, synthetic=False):
    """Load (or build and cache) the labelled feature frame for every subject.

    ``cfg`` is a data-configuration dict (see :func:`sentinel_fl.config.data_config`). With
    ``synthetic=True`` a tiny generated fixture is returned for software tests; real runs never
    fall back to synthetic data. Returns ``(frame, provenance_manifest)``.
    """
    if synthetic:
        df = synthetic_fixture(cfg=cfg)
        return df, dict(
            source="EXPLICIT_SYNTHETIC_FIXTURE",
            synthetic=True,
            fingerprint=hash_frame(df),
            context=dict(enabled=False),
            sleep_label_alignment=cfg.get("sleep_label_position", "start"),
        )
    wes = [Path(cfg["wesad_dir"]) / s / f"{s}.pkl" for s in cfg["wesad_subjects"]]
    missing = [str(p) for p in wes if not p.is_file()]
    labs = sorted(Path(cfg["sleep_label_dir"]).glob("*_labeled_sleep.txt"))
    if missing or not labs:
        raise FileNotFoundError(
            "Real datasets are missing. Point --wesad-dir / --sleep-dir (or the data configuration) "
            "at the dataset folders; no synthetic fallback is used for real runs. "
            f"Missing examples: {missing[:2]}; sleep label files={len(labs)}"
        )
    files = list(wes) + list(labs)
    for p in labs:
        sid = p.name.replace("_labeled_sleep.txt", "")
        for folder, suffix in [
            ("sleep_heart_dir", "heartrate"),
            ("sleep_motion_dir", "acceleration"),
            ("sleep_steps_dir", "steps"),
        ]:
            q = Path(cfg[folder]) / f"{sid}_{suffix}.txt"
            if q.exists():
                files.append(q)
            elif suffix != "steps":
                raise FileNotFoundError(q)
    sig = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(set(files))]
    key = digest(dict(version=VERSION, implementation=code_fingerprint(), config=cfg, sources=sig))
    cache = Path(cfg["cache_dir"]) / f"features_{key[:20]}.pkl"
    if cache.exists():
        with cache.open("rb") as f:
            saved = pickle.load(f)
        if saved["manifest"]["cache_key"] != key:
            raise ValueError("Cache signature mismatch")
        df = validate_frame(saved["data"], cfg.get("context_max_lookahead_s", 0.0))
        if hash_frame(df) != saved["manifest"]["fingerprint"]:
            raise ValueError("Cache content changed")
        return df, saved["manifest"]
    parts, log = [], []
    for p in wes:
        d, entry = load_wesad(p, cfg)
        parts.append(d)
        log.append(entry)
        print(f"WESAD {p.stem}: {len(d)} accepted windows")
    for p in labs:
        sid = p.name.replace("_labeled_sleep.txt", "")
        d, entry = load_sleep(sid, cfg)
        parts.append(d)
        log.append(entry)
        print(f"SLEEP {sid}: {len(d)} accepted epochs")
    if cfg["strict_subjects"] and any(v["kept"] == 0 for v in log):
        raise ValueError("A subject has no usable windows; inspect extraction rather than silently exclude.")
    raw = validate_frame(pd.concat(parts, ignore_index=True), 0.0)
    enriched, context_report = add_context_features(raw, cfg)
    df = validate_frame(enriched, cfg.get("context_max_lookahead_s", 0.0))
    manifest = dict(
        source="raw",
        synthetic=False,
        cache_key=key,
        config=cfg,
        source_signatures=sig,
        fingerprint=hash_frame(df),
        extraction_log=log,
        sleep_label_alignment=cfg["sleep_label_position"],
        context=context_report,
        n_features=int(sum(1 for c in df.columns if c.split("_")[0] in SENSORS)),
    )
    atomic_pickle(cache, dict(data=df, manifest=manifest))
    return df, manifest
