import numpy as np
import pytest
from dataclasses import replace

from sentinel_fl.config import PRESETS
from sentinel_fl.personalization import fit_personal_bias, loo_bias_is_helpful, shrunk_personal_bias
from sentinel_fl.security import CountSketch, aggregate_updates, attack_update, audit_clients, geometric_median, screen
from sentinel_fl.utils import rng_for

RNG = rng_for("test-security", 0)


def test_sketch_is_linear():
    S = CountSketch(1000, 64, 123)
    a, b = RNG.normal(size=1000), RNG.normal(size=1000)
    assert np.allclose(S(2 * a - b), 2 * S(a) - S(b), atol=1e-12)


def _setup():
    S = CountSketch(1000, 64, 123)
    a, b = RNG.normal(size=1000), RNG.normal(size=1000)
    payloads = {"a": a, "b": b}
    sketches = {k: S(v).astype(np.float32).astype(float) for k, v in payloads.items()}
    return S, payloads, sketches, max(np.linalg.norm(a), np.linalg.norm(b))


def test_honest_clients_pass_the_audit():
    S, payloads, sketches, clip = _setup()
    bad, _, _ = audit_clients(S, payloads, sketches, clip)
    assert not bad


def test_inconsistent_client_is_excluded_individually():
    S, payloads, sketches, clip = _setup()
    forged = dict(payloads, a=-payloads["a"])
    bad, _, _ = audit_clients(S, forged, sketches, clip)
    assert bad.get("a") == "inconsistent_sketch" and "b" not in bad


def test_oversized_declared_sketch_cannot_inflate_tolerance():
    S, payloads, sketches, clip = _setup()
    huge = dict(sketches, a=sketches["a"] * 1e9)
    bad, _, _ = audit_clients(S, payloads, huge, clip)
    assert "a" in bad


def test_colluding_equal_and_opposite_discrepancies_are_caught_per_client():
    S, payloads, sketches, clip = _setup()
    w = RNG.normal(size=1000)
    w /= np.linalg.norm(w)
    canceled = {"a": payloads["a"] + w, "b": payloads["b"] - w}
    bad, _, _ = audit_clients(S, canceled, sketches, clip)
    assert set(bad) == {"a", "b"}


def test_screening_downweights_a_reversed_update():
    cfg = replace(PRESETS["smoke"], screen_warmup=0)
    base = RNG.normal(size=64)
    sketches = {str(i): base + 0.01 * RNG.normal(size=64) for i in range(7)}
    sketches["0"] = -5 * base
    trust, diag = screen(sketches, cfg, 1)
    assert diag["0"]["flagged"] and trust["0"] < 1.0


def test_identical_honest_updates_are_retained():
    cfg = replace(PRESETS["smoke"], screen_warmup=0)
    base = RNG.normal(size=64)
    _, diag = screen({str(i): base.copy() for i in range(7)}, cfg, 1)
    assert not any(v["flagged"] for v in diag.values())


def test_screen_cos_gap_can_bind():
    from sentinel_fl.config import Config

    c = Config()
    assert c.screen_cos_gap > c.screen_z * c.screen_mad_floor


def test_support_normalised_mean_does_not_dilute_available_clients():
    out = aggregate_updates([[2.0, 0.0], [4.0, 6.0]], [1, 1], [[True, False], [True, True]])
    assert np.allclose(out, [3.0, 6.0])


def test_geometric_median_reports_convergence():
    gm, conv, _ = geometric_median(RNG.normal(size=(9, 5)), return_status=True)
    assert isinstance(conv, bool) and np.isfinite(gm).all()


def test_simulated_attack_payloads_differ():
    u = RNG.normal(size=20)
    assert not np.allclose(attack_update("sign_flip", u, RNG), attack_update("sketch_forgery", u, RNG))
    with pytest.raises(ValueError):
        attack_update("unknown", u, RNG)


def test_personal_bias_is_bounded_and_direction_correct():
    assert abs(fit_personal_bias(np.full(12, -1.0), np.zeros(12), 5.0)) < 3.0 - 1e-6
    shift = RNG.normal(0, 1, 40) + 3.0
    y = np.ones(40, int)
    y[:12] = 0
    raw = fit_personal_bias(shift, y, 5.0)
    assert raw < 0.0
    assert abs(shrunk_personal_bias(shift, y, 5.0, cap=0.2)) <= 0.2 + 1e-12
    assert abs(shrunk_personal_bias(shift, y, 5.0)) < abs(raw) + 1e-12


def test_leave_one_out_gate_returns_finite_diagnostics():
    z = RNG.normal(0, 1, 24)
    y = (RNG.random(24) < 0.5).astype(int)
    _, base, adapted = loo_bias_is_helpful(z, y, 5.0)
    assert np.isfinite(base) and np.isfinite(adapted)
