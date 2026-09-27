"""Exercise the real-data loaders on a tiny generated dataset written in the original file formats."""

import pickle
from dataclasses import replace

import numpy as np
import pandas as pd

from sentinel_fl.config import PRESETS, data_config
from sentinel_fl.features import load_data
from sentinel_fl.pipeline import run_loso
from sentinel_fl.reporting import write_summary


def _write_wesad(root, subjects, seed=0):
    rng = np.random.default_rng(seed)
    for i, sid in enumerate(subjects):
        n = 700 * 300
        label = np.ones(n, int)
        label[700 * 100 : 700 * 200] = 2
        sig = {
            "wrist": {
                "ACC": rng.normal(0, 1, (32 * 300, 3)) * (1 + (i % 2)),
                "BVP": np.sin(np.arange(64 * 300) * 2 * np.pi * 1.2 / 64) + rng.normal(0, 0.1, 64 * 300),
                "EDA": 1 + 0.01 * np.cumsum(rng.normal(0, 1, 4 * 300)),
                "TEMP": 33 + 0.01 * np.cumsum(rng.normal(0, 1, 4 * 300)),
            },
            "chest": {
                "ACC": rng.normal(0, 1, (n, 3)),
                "ECG": np.sin(np.arange(n) * 2 * np.pi * 1.1 / 700) ** 15 + rng.normal(0, 0.02, n),
                "EDA": 2 + 0.01 * np.cumsum(rng.normal(0, 1, n)),
                "EMG": rng.normal(0, 1, n),
                "Resp": np.sin(np.arange(n) * 2 * np.pi * 0.25 / 700) + rng.normal(0, 0.05, n),
                "Temp": 34 + rng.normal(0, 0.05, n),
            },
        }
        folder = root / sid
        folder.mkdir(parents=True)
        with open(folder / f"{sid}.pkl", "wb") as f:
            pickle.dump(dict(label=label, signal=sig, subject=sid), f)


def _write_sleep(root, subjects, seed=1):
    rng = np.random.default_rng(seed)
    for name in ("heart_rate", "labels", "motion", "steps"):
        (root / name).mkdir(parents=True)
    for sid in subjects:
        epochs = 90
        stage = np.array([0] * 15 + [2] * 60 + [0] * 5 + [2] * 10)
        t = np.arange(epochs) * 30.0
        pd.DataFrame(dict(t=t, s=stage)).to_csv(
            root / "labels" / f"{sid}_labeled_sleep.txt", sep=" ", header=False, index=False
        )
        ta = np.arange(0, epochs * 30, 0.5)
        active = np.repeat(stage == 0, 60)[: len(ta)]
        acc = rng.normal(0, 1, (len(ta), 3)) * (0.02 + 0.5 * active)[:, None]
        pd.DataFrame(np.column_stack([ta, acc])).to_csv(
            root / "motion" / f"{sid}_acceleration.txt", sep=" ", header=False, index=False
        )
        th = np.arange(0, epochs * 30, 5.0)
        hr = 55 + 10 * np.repeat(stage == 0, 6)[: len(th)] + rng.normal(0, 2, len(th))
        pd.DataFrame(dict(t=th, hr=hr)).to_csv(
            root / "heart_rate" / f"{sid}_heartrate.txt", sep=",", header=False, index=False
        )
        ts = np.arange(0, epochs * 30, 60.0)
        pd.DataFrame(dict(t=ts, n=rng.integers(0, 20, len(ts)))).to_csv(
            root / "steps" / f"{sid}_steps.txt", sep=" ", header=False, index=False
        )


def test_real_format_loading_caching_and_a_full_run(tmp_path):
    subjects = [f"S{i}" for i in range(2, 8)]
    _write_wesad(tmp_path / "wesad", subjects)
    _write_sleep(tmp_path / "sleep", [f"{1000 + i}" for i in range(6)])
    dcfg = data_config(
        tmp_path / "wesad", tmp_path / "sleep", wesad_subjects=subjects, cache_dir=str(tmp_path / "cache")
    )
    df, prov = load_data(dcfg)
    assert not prov["synthetic"] and set(df.dataset) == {"SLEEP", "WESAD"}
    assert df.groupby("dataset").client_id.nunique().to_dict() == {"SLEEP": 6, "WESAD": 6}
    assert prov["n_features"] > 50 and prov["context"]["max_declared_lookahead_s"] > 0
    df2, prov2 = load_data(dcfg)  # served from the cache
    assert prov2["fingerprint"] == prov["fingerprint"] and len(df2) == len(df)

    cfg = replace(PRESETS["smoke"], hidden=6, rounds=2, hpo_rounds=1, sequence_decode=True, byte_budget=3000)
    metrics, systems, personal, ledger, manifest = run_loso(df, prov, cfg, tmp_path / "out", data_config=dcfg)
    assert ledger.status.eq("completed").all() and len(metrics) == 2
    summary, status = write_summary(tmp_path / "out")
    assert status["complete"]
    assert (tmp_path / "out" / "RESULTS_SUMMARY.md").exists()
    assert metrics.macro_f1.between(0, 1).all()
