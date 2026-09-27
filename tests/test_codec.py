import numpy as np
import pytest

from sentinel_fl.codec import (
    PACKET_HEADER,
    ByteCodec,
    ErrorMemory,
    InfeasibleBudget,
    bounded_allocate,
    make_packet,
    make_record,
    quantize_values,
)
from sentinel_fl.utils import rng_for


def _vector(n=1000, seed=0):
    rng = rng_for("test-codec", seed)
    return rng.normal(size=n) * rng.lognormal(0, 0.8, n)


def _codec(n=1000):
    return ByteCodec([("large", 0, 700), ("small", 700, n)], n)


@pytest.mark.parametrize("bits", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("keep", [20, 500, 1000])
def test_serialized_record_round_trips(bits, keep):
    v = _vector()
    idx = np.argsort(-abs(v))[:keep]
    packet, rec = make_record(v, 65535, bits, idx)
    assert np.array_equal(_codec().decode(make_packet(1000, [packet])), rec)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_quantizer_uses_full_alphabet(bits):
    _, q, _ = quantize_values(np.linspace(-1, 1, 4096), bits)
    assert len(np.unique(q)) == 2**bits


@pytest.mark.parametrize("budget", [100, 200, 400, 800, 1600, 4000, 5000])
def test_hard_byte_cap_and_exact_error(budget):
    v, codec = _vector(), _codec()
    enc = codec.encode(v, budget)
    assert enc["bytes"] <= budget
    assert np.isclose(enc["error"], np.sum((v - codec.decode(enc["packet"])) ** 2))


def test_distortion_is_monotone_in_budget():
    v, codec = _vector(), _codec()
    errs = [codec.encode(v, b)["error"] for b in (100, 200, 400, 800, 1600, 4000, 5000)]
    assert np.all(np.diff(errs) <= 1e-9)


def test_unaffordable_budget_raises_instead_of_sending_nothing():
    with pytest.raises(InfeasibleBudget):
        _codec().sparse_packet(_vector(), PACKET_HEADER.size + 1)


def test_dense_mode_is_lossless_up_to_fp32():
    v, codec = _vector(), _codec()
    enc = codec.encode(v, 10_000, "dense")
    assert np.allclose(enc["recon"], v, rtol=1e-6, atol=1e-6)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        _codec().encode(_vector(), 5000, "unknown")


def test_truncated_packet_fails_closed():
    codec = _codec()
    with pytest.raises(ValueError):
        codec.decode(codec.dense(_vector())[0][:-1])


def test_error_feedback_telescopes():
    codec, memory = _codec(), ErrorMemory(1000)
    rng = rng_for("test-ef", 0)
    true, sent = np.zeros(1000), np.zeros(1000)
    for _ in range(12):
        u = rng.normal(0, 0.1, 1000)
        target = memory.target(u)
        enc = codec.encode(target, 600)
        true += u
        sent += enc["recon"]
        memory.acknowledge(target, enc["recon"])
    assert np.allclose(true - sent, memory.residual, atol=1e-12)


def test_rollback_keeps_the_whole_update_owed():
    memory = ErrorMemory(50)
    target = memory.target(rng_for("test-rb", 0).normal(0, 0.1, 50))
    memory.rollback(target)
    assert np.allclose(memory.residual, target, atol=1e-15)


@pytest.mark.parametrize("total", [300, 450, 999])
def test_allocation_respects_caps_and_total(total):
    caps = {"a": 100, "b": 350, "c": 600}
    alloc = bounded_allocate(caps, total, 100, {"a": 10, "b": 1, "c": 0.01})
    assert sum(alloc.values()) == total and all(100 <= alloc[k] <= caps[k] for k in alloc)


def test_zero_fleet_with_zero_floor_is_feasible():
    assert bounded_allocate({"a": 0, "b": 0}, 0, 0) == {"a": 0, "b": 0}
