import numpy as np
from sklearn.metrics import f1_score

from sentinel_fl.config import PRESETS
from sentinel_fl.decoding import SequenceDecoder
from sentinel_fl.metrics import bootstrap_mean, metric_dict
from sentinel_fl.model import ResidualModel, train_local
from sentinel_fl.utils import rng_for


def test_gradients_match_finite_differences():
    rng = rng_for("test-grad", 0)
    model = ResidualModel(4, 3, 12)
    theta = model.flat() + rng.normal(0, 0.1, model.size)
    model.set_flat(theta)
    X = rng.normal(size=(8, 4))
    y = np.array([0, 1] * 4)
    cw = np.array([0.8, 1.2])
    _, analytic = model.loss_gradient(X, y, cw, theta - 0.1, 0.03, 0.005)
    numeric, eps = [], 1e-6
    for k in range(model.size):
        a = theta.copy()
        a[k] += eps
        model.set_flat(a)
        lp = model.loss_gradient(X, y, cw, theta - 0.1, 0.03, 0.005)[0]
        a[k] -= 2 * eps
        model.set_flat(a)
        lm = model.loss_gradient(X, y, cw, theta - 0.1, 0.03, 0.005)[0]
        numeric.append((lp - lm) / (2 * eps))
    assert np.max(np.abs(analytic - numeric)) < 2e-6


def test_local_training_keeps_no_state_between_clients():
    rng = rng_for("test-iso", 0)
    model = ResidualModel(4, 3, 12)
    theta = model.flat() + rng.normal(0, 0.1, model.size)
    X = rng.normal(size=(8, 4))
    y = np.array([0, 1] * 4)
    cw = np.array([0.8, 1.2])
    cfg = PRESETS["smoke"]
    a, _ = train_local(model, theta, X, y, cw, cfg, rng_for("iso"), 0.01)
    train_local(model, theta, X * 3, 1 - y, cw, cfg, rng_for("other"), 0.01)
    b, _ = train_local(model, theta, X, y, cw, cfg, rng_for("iso"), 0.01)
    assert np.array_equal(a, b)


def test_unsupported_parameters_receive_no_update():
    rng = rng_for("test-support", 0)
    model = ResidualModel(4, 3, 1)
    available = np.array([True, True, False, True])
    support = model.parameter_support(available)
    X = rng.normal(size=(16, 4))
    y = rng.integers(0, 2, 16)
    delta, _ = train_local(
        model, model.flat(), X, y, np.array([1.0, 1.0]), PRESETS["smoke"], rng_for("s"), 0.0, support
    )
    assert np.all(delta[~support] == 0)


def _chain(n_subjects=3, epochs=120):
    import pandas as pd

    rng = rng_for("test-chain", 0)
    rows = []
    for c in range(n_subjects):
        st = 0
        for t in range(epochs):
            if rng.random() < 0.08:
                st = 1 - st
            rows.append(dict(client_id=f"S{c}", label_start=t * 30.0, y=st))
    return pd.DataFrame(rows)


def test_sequence_decoder_transition_matrix_and_noop():
    df = _chain()
    dec = SequenceDecoder((0.0, 0.5, 1.0)).fit(df)
    assert np.allclose(dec.A.sum(axis=1), 1.0)
    assert dec.A[0, 0] > 0.5 and dec.A[1, 1] > 0.5
    p = np.where(df.y.to_numpy() == 1, 0.8, 0.2)
    assert np.allclose(dec.smooth(df, p, 0.0), p, atol=1e-12)


def test_sequence_decoder_helps_on_a_noisy_persistent_chain():
    df = _chain()
    dec = SequenceDecoder((0.0, 0.5, 1.0)).fit(df)
    rng = rng_for("noise", 0)
    noisy = np.clip(np.where(df.y.to_numpy() == 1, 0.8, 0.2) + rng.normal(0, 0.22, len(df)), 0.02, 0.98)
    y = df.y.to_numpy(int)
    base = f1_score(y, (noisy >= 0.5).astype(int), average="macro", zero_division=0)
    got = f1_score(y, (dec.smooth(df, noisy, 1.0) >= 0.5).astype(int), average="macro", zero_division=0)
    assert got >= base


def test_metrics_hand_calculation_and_degenerate_flag():
    m = metric_dict([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert abs(m["probability_ece"] - 0.15) < 1e-12 and abs(m["confidence_ece"] - 0.15) < 1e-12
    one = metric_dict([0, 0], [0.1, 0.1])
    assert one["degenerate_subject"] and np.isnan(one["auroc"]) and one["macro_f1"] == 0.5


def test_bootstrap_refuses_underpowered_interval():
    assert all(np.isnan(v) for v in bootstrap_mean([0.1, 0.2, 0.3], 200, "t", 6))
