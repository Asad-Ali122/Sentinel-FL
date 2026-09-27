import numpy as np
import pandas as pd
import pytest

from sentinel_fl.features import add_context_features, complete_steps, label_gate, validate_frame
from sentinel_fl.preprocessing import (
    Preprocessor,
    assert_temporal_disjoint,
    audit_partitions,
    make_splits,
    role_frame,
    support_query,
)
from dataclasses import replace


def test_loso_holds_every_subject_once_and_roles_are_disjoint(fixture_frame, smoke_cfg):
    splits = make_splits(fixture_frame, replace(smoke_cfg, max_folds=None))
    assert len(splits) == fixture_frame.client_id.nunique()
    assert len({s["held"] for s in splits}) == len(splits)
    assert all(audit_partitions(s) for s in splits)


def test_participant_overlap_is_rejected(fixture_frame, smoke_cfg):
    sp = dict(make_splits(fixture_frame, smoke_cfg)[0])
    sp["val"] = sp["test"]
    with pytest.raises(AssertionError):
        audit_partitions(sp)


def test_preprocessor_is_fit_on_fit_rows_only(fixture_frame, smoke_cfg):
    sp = make_splits(fixture_frame, smoke_cfg)[0]
    prep = Preprocessor().fit(role_frame(fixture_frame, sp, "fit"))
    before = prep.med.copy()
    prep.transform(role_frame(fixture_frame, sp, "test").assign(acc_f0=1e10))
    assert np.array_equal(before, prep.med)


def test_support_query_is_temporally_purged(fixture_frame, smoke_cfg):
    sp = make_splits(fixture_frame, smoke_cfg)[0]
    support, query = support_query(role_frame(fixture_frame, sp, "test"), 10)
    assert len(query) > 0 and assert_temporal_disjoint(support, query)


def _demo(seed=0, subjects=3, epochs=120):
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(subjects):
        st = 0
        for t in range(epochs):
            if rng.random() < 0.08:
                st = 1 - st
            rows.append(
                dict(
                    dataset="SLEEP",
                    task="sleep",
                    client_id=f"SLEEP_D{cid}",
                    y=st,
                    feature_start=t * 30 - 60,
                    feature_end=t * 30 + 30,
                    label_start=t * 30,
                    label_end=t * 30 + 30,
                    label_coverage=1.0,
                    label_purity=1.0,
                    acc_watch_mag_mean=float(st * 2 + rng.normal()),
                    acc_watch_mag_std=float(abs(rng.normal())),
                    hr_epoch_mean=float(60 + st * 8 + rng.normal(0, 3)),
                )
            )
    return pd.DataFrame(rows)


CTX = dict(
    context_enabled=True,
    context_centered=True,
    context_subject_relative=True,
    context_time_features=True,
    context_max_lookahead_s=600.0,
    context_scales_by_dataset=dict(SLEEP=(4, 12)),
    context_forward_by_dataset=dict(SLEEP=True),
    context_base_columns=dict(SLEEP=("acc_watch_mag_mean", "hr_epoch_mean")),
)


def test_context_features_declare_their_lookahead():
    enriched, report = add_context_features(_demo(), CTX)
    assert report["n_context_features"] > 10
    assert np.isclose(enriched.declared_lookahead_s.max(), 12 * 30 / 2)
    assert validate_frame(enriched, 600.0) is not None
    with pytest.raises(ValueError):
        validate_frame(enriched, 0.0)
    cols = [c for c in enriched if c.startswith(("ctx_", "time_"))]
    assert np.isfinite(enriched[cols].to_numpy(float)).all()


def test_subject_relative_features_ignore_other_subjects():
    demo = _demo()
    a, _ = add_context_features(demo, CTX)
    shifted = demo.copy()
    m = shifted.client_id == "SLEEP_D0"
    shifted.loc[m, "hr_epoch_mean"] += 1000.0
    b, _ = add_context_features(shifted, CTX)
    col = "ctx_hr_epoch_mean_z"
    assert np.allclose(a[a.client_id == "SLEEP_D1"][col], b[b.client_id == "SLEEP_D1"][col], atol=1e-12)


def test_causal_time_mode_adds_no_recording_length_features():
    enriched, _ = add_context_features(_demo(), dict(CTX, context_time_mode="causal"))
    assert not any(c.startswith("time_r_") for c in enriched)


def test_label_gate_and_steps_semantics():
    gate, cov, pur = label_gate([0] * 9 + [2], {1, 2, 3})
    assert gate is None and cov == 0.1 and pur == 0.1
    steps = np.array([[0, 100], [100, 200], [200, 300]], float)
    assert np.array_equal(complete_steps(steps, -10, 150), [100])
    assert np.array_equal(complete_steps(steps, -10, 1000), [100, 200])
