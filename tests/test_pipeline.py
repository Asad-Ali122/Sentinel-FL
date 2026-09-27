import pickle
from dataclasses import replace

import numpy as np
import pandas as pd

from sentinel_fl import SentinelFLPredictor
from sentinel_fl.cli import main
from sentinel_fl.federated import LR_SEARCH, SENTINEL_FL, Scenario, run_federated
from sentinel_fl.pipeline import run_fold
from sentinel_fl.preprocessing import Preprocessor, make_splits, role_frame


def _fold(df, cfg, dataset="WESAD"):
    cohort = df[df.dataset == dataset].copy()
    sp = make_splits(cohort, cfg)[0]
    return cohort, sp, role_frame(cohort, sp, "fit"), role_frame(cohort, sp, "val")


def test_completed_checkpoint_resumes_identically(tmp_path, fixture_frame, smoke_cfg):
    cohort, sp, fit, val = _fold(fixture_frame, smoke_cfg)
    prep = Preprocessor().fit(fit)
    cfg = replace(smoke_cfg, rounds=2, eval_every=1)
    path = tmp_path / "resume.pkl"
    m1, s1 = run_federated(fit, val, prep, cfg, SENTINEL_FL, Scenario(), 13, path, "test")
    m2, s2 = run_federated(fit, val, prep, cfg, SENTINEL_FL, Scenario(), 13, path, "test")
    assert np.array_equal(m1.flat(), m2.flat()) and s1["uplink"] == s2["uplink"]
    assert np.isfinite(m1.logits(prep.transform(val))).all()


def test_every_upload_stays_within_its_cap(fixture_frame, smoke_cfg):
    cohort, sp, fit, val = _fold(fixture_frame, smoke_cfg)
    prep = Preprocessor().fit(fit)
    _, state = run_federated(fit, val, prep, smoke_cfg, SENTINEL_FL, Scenario(), 13)
    assert all(e["allocated_bytes"] <= e["physical_cap"] for e in state["events"])
    assert state["uplink"] > 0


def test_forged_sketch_excludes_the_forger_but_not_the_round(fixture_frame, smoke_cfg):
    cohort, sp, fit, val = _fold(fixture_frame, smoke_cfg)
    prep = Preprocessor().fit(fit)
    cfg = replace(smoke_cfg, rounds=3, eval_every=1, screen_warmup=0)
    forge = Scenario(name="sketch_forgery", attack="sketch_forgery", bad_fraction=0.34)
    _, state = run_federated(fit, val, prep, cfg, SENTINEL_FL, forge, 13)
    logs = pd.DataFrame(state["logs"])
    assert (logs.status != "all_audit_excluded").any() and logs.participants.max() > 0
    assert logs.audit_excluded.sum() > 0


def test_dense_search_protocol_sends_no_sketch(fixture_frame, smoke_cfg):
    cohort, sp, fit, val = _fold(fixture_frame, smoke_cfg)
    prep = Preprocessor().fit(fit)
    _, state = run_federated(fit, val, prep, replace(smoke_cfg, rounds=1), LR_SEARCH, Scenario(), 13)
    assert pd.DataFrame(state["logs"]).sketch_bytes.fillna(0).sum() == 0


def test_run_fold_artifact_reproduces_saved_predictions(tmp_path, fixture_frame, smoke_cfg):
    cohort = fixture_frame[fixture_frame.dataset == "SLEEP"].copy()
    sp = make_splits(cohort, smoke_cfg)[0]
    result = run_fold(cohort, sp, smoke_cfg, Scenario(), 13, tmp_path, "test", True)
    saved = pd.read_csv(result["run_dir"] + "/predictions.csv")
    predictor = SentinelFLPredictor.load(result["run_dir"] + "/artifact.pkl")
    test = role_frame(cohort, sp, "test")
    assert np.allclose(predictor.predict_proba(test), saved.p.to_numpy(), atol=1e-12)
    assert set(np.unique(predictor.predict(test))) <= {0, 1}
    again = run_fold(cohort, sp, smoke_cfg, Scenario(), 13, tmp_path, "test", True)
    assert again["metrics"]["macro_f1"] == result["metrics"]["macro_f1"]
    with open(result["run_dir"] + "/result.pkl", "rb") as f:
        assert pickle.load(f)["metrics"]["client_id"] == sp["held"]


def test_cli_smoke_run_writes_summary(tmp_path):
    main(["run", "--mode", "smoke", "--out", str(tmp_path)])
    for name in (
        "participant_metrics.csv",
        "system_metrics.csv",
        "completion_ledger.csv",
        "summary_by_dataset.csv",
        "RESULTS_SUMMARY.md",
        "run_manifest.json",
        "run_status.json",
    ):
        assert (tmp_path / name).exists(), name
    text = (tmp_path / "RESULTS_SUMMARY.md").read_text()
    assert "Synthetic fixture data: YES" in text
    assert pd.read_csv(tmp_path / "completion_ledger.csv").status.eq("completed").all()
