"""Data configuration, hyper-parameters and run presets for SENTINEL-FL."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

__version__ = "2.1.0"
VERSION = "SENTINEL-FL-2.1"

# Sensor groups recognised by the feature pipeline. "ctx" and "time" are derived
# pseudo-sensor groups produced by ``add_context_features``; the Preprocessor
# discovers model inputs by this prefix list.
SENSORS = ("acc", "ppg", "hr", "ecg", "eda", "emg", "resp", "temp", "steps", "ctx", "time")

# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------
DEFAULT_DATA = dict(
    # WESAD: <wesad_dir>/S2/S2.pkl, <wesad_dir>/S3/S3.pkl, ...
    wesad_dir="data/WESAD",
    wesad_subjects=["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"],
    # PhysioNet "Motion and heart rate from a wrist-worn wearable and labeled sleep from PSG" layout.
    sleep_heart_dir="data/sleep/heart_rate",
    sleep_label_dir="data/sleep/labels",
    sleep_motion_dir="data/sleep/motion",
    sleep_steps_dir="data/sleep/steps",
    wesad_window_s=60.0,
    wesad_stride_s=30.0,
    sleep_epoch_s=30.0,
    sleep_context_s=120.0,
    steps_context_s=300.0,
    # A label timestamp starts its scored epoch; predictions are issued at the epoch END.
    # If your export timestamps END epochs, set this to "end".
    sleep_label_position="start",
    min_valid_coverage=0.95,
    min_label_purity=0.95,
    include_amusement_as_nonstress=True,
    sensor_profile="all",  # "all" (wrist + chest) or "wrist" (wrist sensors only)
    cache_dir="sentinel_fl_cache",
    strict_subjects=True,
    # ---- epoch-sequence context features (computed inside each subject's own recording) ----
    context_enabled=True,
    context_scales=(4, 12, 40),  # epochs; with 30 s epochs this is 2 / 6 / 20 minutes
    context_centered=True,  # centered (retrospective) rolling windows
    context_subject_relative=True,  # within-recording robust z-scores and percentile ranks
    context_time_features=True,
    # "causal": elapsed-time features that are computable online at epoch t.
    # "retrospective": additionally derives features from the recording's total length
    # (fraction elapsed, minutes remaining), which needs the recording end and is therefore offline only.
    context_time_mode="causal",
    # Maximum look-ahead (seconds) that any feature may use. It is enforced per row by ``validate_frame``
    # and written to the provenance manifest.
    context_max_lookahead_s=600.0,
    context_scales_by_dataset=dict(SLEEP=(4, 12, 40), WESAD=(2, 6)),
    # Forward-looking context is legitimate for retrospective sleep scoring, not for an online stress detector.
    context_forward_by_dataset=dict(SLEEP=True, WESAD=False),
    context_base_columns=dict(
        SLEEP=(
            "acc_watch_mag_mean",
            "acc_watch_mag_std",
            "acc_watch_activity",
            "acc_watch_a0_std",
            "hr_epoch_mean",
            "hr_epoch_std",
            "hr_epoch_dmean",
            "hr_context_mean",
            "steps_complete_mean",
        ),
        WESAD=(
            "acc_wrist_mag_mean",
            "acc_wrist_activity",
            "eda_wrist_mean",
            "eda_wrist_slope",
            "temp_wrist_mean",
            "ppg_wrist_pulse_rate_median",
            "ecg_chest_pulse_rate_median",
            "resp_chest_mean",
        ),
    ),
)


@dataclass(frozen=True)
class Config:
    """Hyper-parameters of a SENTINEL-FL run (federated protocol, calibration, decoding, personalisation)."""

    # --- federated optimisation ---------------------------------------------
    rounds: int = 80
    local_steps: int = 10
    batch_size: int = 128
    hidden: int = 48
    lr: float = 0.003
    weight_decay: float = 1e-4
    prox_mu: float = 0.01
    server_lr: float = 1.0
    server_momentum: float = 0.5
    clients_per_round: int = 10
    clip_norm: float = 1.0
    feature_clip: float = 8.0
    class_weight_power: float = 1.0  # 1.0 = full inverse-frequency class weighting
    val_subjects: int = 4
    eval_every: int = 5
    # --- sketch screening / audit ---------------------------------------------
    sketch_dim: int = 128
    screen_min_clients: int = 5
    screen_z: float = 4.5
    screen_cos_floor: float = -0.25
    # Must exceed screen_z * screen_mad_floor (0.45 with the defaults) for the conjunct to be able to bind.
    screen_cos_gap: float = 0.55
    screen_log_norm_gap: float = 1.25
    screen_min_weight: float = 0.1
    screen_warmup: int = 3
    screen_mad_floor: float = 0.1
    # --- byte codec --------------------------------------------------------------
    byte_budget: int = 8_000  # hard cap of one client upload in bytes (envelope + sketch included)
    device_min_bytes: int = 1_000
    encoding_keep: tuple = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
    encoding_bits: tuple = (2, 4, 8, 16, 32)
    # --- evaluation protocol -------------------------------------------------------
    seeds: tuple = (13, 42, 73)
    max_folds: int | None = None  # None = every subject is held out once
    hp_lrs: tuple = (0.001, 0.003, 0.01)  # learning-rate grid, tuned on validation subjects
    hpo_rounds: int = 25
    # --- sequence decoding and personalisation --------------------------------------
    sequence_decode: bool = True
    sequence_strengths: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    sequence_min_gap_ratio: float = 1.75  # epochs further apart than this break a run
    adapt_shots: int = 20
    adapt_ridge: float = 5.0
    adapt_require_both_classes: bool = True
    adapt_gate_by_loo: bool = True
    adapt_loo_margin: float = 0.03
    adapt_min_minority: int = 4
    adapt_max_bias: float = 1.0
    adapt_min_logit_sd: float = 0.05
    # --- reporting ---------------------------------------------------------------------
    bootstrap_reps: int = 5000
    min_ci_subjects: int = 6  # participant-bootstrap intervals are only reported from this many subjects
    n_threads: int = 1


PRESETS = {
    # Tiny synthetic software test (runs in seconds on any laptop).
    "smoke": Config(
        rounds=3,
        local_steps=2,
        hidden=8,
        val_subjects=1,
        eval_every=1,
        clients_per_round=5,
        seeds=(13,),
        max_folds=1,
        hp_lrs=(0.003,),
        hpo_rounds=2,
        bootstrap_reps=200,
        byte_budget=1200,
        device_min_bytes=128,
        sequence_decode=False,
    ),
    # Holds out 8 subjects per cohort with one seed.
    "pilot": Config(
        rounds=50, seeds=(13,), max_folds=8, val_subjects=3, hp_lrs=(0.001, 0.003), hpo_rounds=15, bootstrap_reps=1000
    ),
    # Leave-one-subject-out over every subject, three seeds.
    "study": Config(rounds=100),
}


def get_config(mode: str = "study") -> Config:
    """Return the preset hyper-parameters for ``smoke``, ``pilot`` or ``study``."""
    if mode not in PRESETS:
        raise ValueError(f"Unknown mode {mode!r}; choose one of {sorted(PRESETS)}")
    return PRESETS[mode]


def data_config(wesad_dir=None, sleep_dir=None, **overrides) -> dict:
    """Copy of :data:`DEFAULT_DATA` with dataset folders and arbitrary keys overridden.

    ``sleep_dir`` is the folder holding the four sleep sub-folders ``heart_rate``, ``labels``,
    ``motion`` and ``steps``.
    """
    cfg = dict(DEFAULT_DATA)
    if wesad_dir is not None:
        cfg["wesad_dir"] = str(wesad_dir)
    if sleep_dir is not None:
        root = Path(sleep_dir)
        cfg.update(
            sleep_heart_dir=str(root / "heart_rate"),
            sleep_label_dir=str(root / "labels"),
            sleep_motion_dir=str(root / "motion"),
            sleep_steps_dir=str(root / "steps"),
        )
    cfg.update(overrides)
    return cfg


def code_fingerprint() -> str:
    """Hash of the package source files; part of every cache / checkpoint key."""
    h = hashlib.sha256(VERSION.encode())
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()
