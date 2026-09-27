"""Federated training loop: byte-capped client uploads, sketch screening, per-client audit, aggregation."""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .codec import (
    ENVELOPE_HEADER,
    PACKET_HEADER,
    TELEMETRY,
    ByteCodec,
    ErrorMemory,
    InfeasibleBudget,
    bounded_allocate,
)
from .config import VERSION
from .metrics import participant_score
from .model import ResidualModel, l2_clip, train_local
from .preprocessing import class_weights
from .security import CountSketch, aggregate_updates, attack_update, audit_clients, screen
from .utils import atomic_pickle, digest, hash_frame, rng_for


@dataclass(frozen=True)
class Protocol:
    """Transport and defence configuration of one federated run.

    ``codec="sentinel"`` sends serialized, byte-capped packets with client-side error feedback and is
    the SENTINEL-FL protocol. ``codec="dense"`` sends uncapped fp32 updates without error feedback and
    is used only for the learning-rate search. ``screen`` and ``audit`` switch the server-side
    sketch screening and the per-client payload audit (which also make clients send a sketch).
    """

    codec: str = "sentinel"
    screen: bool = True
    audit: bool = True

    @property
    def sends_sketch(self):
        return self.screen or self.audit

    @property
    def error_feedback(self):
        return self.codec == "sentinel"


SENTINEL_FL = Protocol()
LR_SEARCH = Protocol(codec="dense", screen=False, audit=False)


@dataclass(frozen=True)
class Scenario:
    """Deployment conditions of a run.

    ``attack`` / ``bad_fraction`` simulate malicious clients (``sign_flip``, ``scaled``,
    ``gaussian``, ``label_flip``, ``sketch_forgery``, ``free_rider``, ``nullspace``) to exercise the
    screening and audit; ``device_tiers`` assigns heterogeneous per-device byte caps and
    ``fleet_fraction`` limits the total bytes the fleet may upload in a round.
    """

    name: str = "clean"
    attack: str = "clean"
    bad_fraction: float = 0.0
    device_tiers: tuple = ()
    fleet_fraction: float = 1.0


def make_clients(df, prep, model, seed, scenario):
    """Build per-client tensors, capability masks and the (simulated) malicious set."""
    ids = sorted(df.client_id.unique())
    order = rng_for("bad-assignment", seed).permutation(ids)
    nb = int(math.floor(scenario.bad_fraction * len(ids)))
    if scenario.bad_fraction > 0:
        nb = max(1, nb)
    bad = set(order[:nb]) if scenario.attack != "clean" else set()
    clients = {}
    for cid, g in df.groupby("client_id", sort=True):
        X = prep.transform(g)
        y = g.y.to_numpy(int).copy()
        available = prep.available_inputs(g)
        if cid in bad and scenario.attack == "label_flip":
            y = 1 - y
        support = model.parameter_support(available)
        clients[cid] = dict(X=X, y=y, support=support, n=len(y), bad=cid in bad)
    return clients


def transport_overhead(protocol, cfg):
    """Fixed per-upload bytes: envelope header, telemetry and, if used, the sketch."""
    return ENVELOPE_HEADER.size + TELEMETRY.size + (4 * cfg.sketch_dim if protocol.sends_sketch else 0)


def build_envelope(packet, sketch, error, energy, round_index, client_slot):
    """Serialize one client upload: envelope header, update packet, optional sketch, telemetry."""
    sb = b"" if sketch is None else np.asarray(sketch, dtype="<f4").tobytes()
    if not np.isfinite(error) or not np.isfinite(energy):
        raise FloatingPointError("Invalid telemetry")
    return (
        ENVELOPE_HEADER.pack(b"WENV", round_index, client_slot, len(packet), len(sb))
        + packet
        + sb
        + TELEMETRY.pack(error, energy)
    )


def client_transmission(codec, sketcher, target, honest_target, budget, protocol, kind):
    """What a client emits: the packet and the sketch of what it actually transmitted.

    The server never sees ``target``, only the packet and the declared sketch. A simulated
    ``sketch_forgery`` client declares the sketch of an innocuous update instead.
    """
    enc = codec.encode(target, budget, protocol.codec)
    packet = enc["packet"]
    sk = None
    if protocol.sends_sketch:
        declared = enc["recon"]
        if kind == "sketch_forgery":
            declared = codec.encode(honest_target, budget, protocol.codec)["recon"]
        sk = sketcher(declared)
        sk = np.asarray(sk, dtype="<f4").astype(float)
    return packet, sk, enc


def checkpoint_snapshot(state):
    """Round state with error-feedback residuals stored as plain arrays."""
    snap = dict(state)
    snap["memories"] = {k: np.asarray(v.residual, float) for k, v in state["memories"].items()}
    return snap


def restore_checkpoint(old, ids, dim):
    """Rebuild the round state (including per-client :class:`ErrorMemory`) from a snapshot."""
    state = dict(old)
    mem = {}
    for k in ids:
        m = ErrorMemory(dim)
        stored = old.get("memories", {}).get(k)
        if stored is not None:
            m.residual = np.asarray(stored, float).copy()
        mem[k] = m
    state["memories"] = mem
    return state


def _event(r, cid, clients, base_caps, budgets, scenario, *, audit_round_failed, excluded, **extra):
    row = dict(
        round=r,
        client_id=cid,
        malicious=clients[cid]["bad"],
        physical_cap=base_caps[cid],
        allocated_bytes=budgets[cid],
        audit_round_failed=audit_round_failed,
        excluded=excluded,
        attack=scenario.attack if clients[cid]["bad"] else "clean",
    )
    row.update(extra)
    return row


def run_federated(fit, val, prep, cfg, protocol, scenario, seed, checkpoint=None, code_hash="unspecified"):
    """Run the federated protocol over the ``fit`` participants and return ``(model, state)``.

    Every round: sample clients, train locally with a proximal term, add the error-feedback
    residual, encode under the allocated byte cap, decode on the server, audit each client's
    payload against its sketch, screen sketches, aggregate with participant weights normalised by
    coordinate support, apply server momentum, and commit error feedback only for updates that
    were actually applied. The returned model carries the parameters of the best validation round.
    If ``checkpoint`` is given, progress is saved there and a matching checkpoint is resumed.
    """
    model = ResidualModel(prep.dimension, cfg.hidden, seed)
    initial = model.flat()
    codec = ByteCodec(model.blocks, model.size, cfg.encoding_keep, cfg.encoding_bits)
    clients = make_clients(fit, prep, model, seed, scenario)
    ids = sorted(clients)
    cw = class_weights(fit, cfg.class_weight_power)
    Xval = prep.transform(val)
    contract = digest(
        dict(
            version=VERSION,
            code=code_hash,
            data=hash_frame(fit),
            val=hash_frame(val),
            config=asdict(cfg),
            protocol=asdict(protocol),
            scenario=asdict(scenario),
            seed=seed,
            features=prep.features,
        )
    )
    state = dict(
        contract=contract,
        round=0,
        theta=initial,
        momentum=np.zeros(model.size),
        memories={k: ErrorMemory(model.size) for k in ids},
        logs=[],
        events=[],
        best_theta=initial.copy(),
        best_score=-np.inf,
        best_round=0,
        uplink=0,
        downlink=0,
        selected_uplink=0,
        selected_downlink=0,
        train_seconds=0.0,
        encode_seconds=0.0,
        screen_seconds=0.0,
        audit_seconds=0.0,
        aggregate_seconds=0.0,
    )
    if checkpoint and Path(checkpoint).exists():
        with open(checkpoint, "rb") as f:
            old = pickle.load(f)
        if old.get("contract") != contract:
            raise ValueError("Checkpoint contract changed; use a new output location")
        state = restore_checkpoint(old, ids, model.size)
    overhead = transport_overhead(protocol, cfg)
    dense_size = len(codec.dense(initial)[0])
    is_dense = protocol.codec == "dense"
    base_caps = {cid: (dense_size + overhead if is_dense else cfg.byte_budget) for cid in ids}
    if scenario.device_tiers and not is_dense:
        ordering = rng_for("device-tiers", seed).permutation(ids).tolist()
        base_caps = {cid: int(scenario.device_tiers[j % len(scenario.device_tiers)]) for j, cid in enumerate(ordering)}
    if min(base_caps.values()) < overhead + PACKET_HEADER.size:
        raise ValueError("A device cannot afford metadata; increase caps or reduce sketch_dim")
    with threadpool_limits(limits=cfg.n_threads):
        for r in range(state["round"] + 1, cfg.rounds + 1):
            start = time.perf_counter()
            theta = state["theta"]
            selected = (
                rng_for("participation", seed, r).permutation(ids)[: min(cfg.clients_per_round, len(ids))].tolist()
            )
            caps = {k: base_caps[k] for k in selected}
            total = min(sum(caps.values()), int(sum(caps.values()) * scenario.fleet_fraction))
            floor = min(cfg.device_min_bytes, min(caps.values()))
            floor = max(floor, overhead + PACKET_HEADER.size)
            budgets = bounded_allocate(caps, total, floor)
            sketcher = CountSketch(model.size, cfg.sketch_dim, seed=f"{seed}:{r}")
            payloads, sketches, supports, pending = {}, {}, {}, {}
            errors, energies = [], []
            packetbytes = sketchbytes = train_steps = 0
            infeasible = []
            t_train = t_encode = t_screen = t_audit = t_aggregate = 0.0
            for cid in selected:
                c = clients[cid]
                t = time.perf_counter()
                if c["bad"] and scenario.attack == "free_rider":
                    update = np.zeros(model.size)
                    honest = np.zeros(model.size)
                else:
                    update, _ = train_local(
                        model, theta, c["X"], c["y"], cw, cfg, rng_for("local", seed, r, cid), cfg.prox_mu, c["support"]
                    )
                    train_steps += cfg.local_steps
                    honest = l2_clip(update, cfg.clip_norm)
                t_train += time.perf_counter() - t
                kind = scenario.attack if c["bad"] else "clean"
                if not (c["bad"] and scenario.attack == "free_rider"):
                    update = attack_update(kind, honest, rng_for("attack", seed, r, cid), sketcher)
                mem = state["memories"][cid]
                if protocol.error_feedback:
                    target = mem.target(update, c["support"])
                else:
                    target = np.where(c["support"], update, 0.0)
                honest_target = np.where(c["support"], honest, 0.0)
                t = time.perf_counter()
                try:
                    packet, sk, enc = client_transmission(
                        codec, sketcher, target, honest_target, budgets[cid] - overhead, protocol, kind
                    )
                except InfeasibleBudget:
                    infeasible.append(cid)
                    if protocol.error_feedback:
                        mem.rollback(target)
                    state["events"].append(
                        _event(
                            r,
                            cid,
                            clients,
                            base_caps,
                            budgets,
                            scenario,
                            audit_round_failed=False,
                            excluded="budget_infeasible",
                            screened=False,
                            flagged=False,
                            weight=0.0,
                        )
                    )
                    t_encode += time.perf_counter() - t
                    continue
                rec = codec.decode(packet)
                error = float(np.sum((target - rec) ** 2))
                energy = float(target @ target)
                env = build_envelope(packet, sk, error, energy, r, ids.index(cid))
                if len(env) > budgets[cid]:
                    raise AssertionError("Actual serialized upload exceeds budget")
                packetbytes += len(env)
                sketchbytes += 0 if sk is None else 4 * len(sk)
                payloads[cid] = rec
                pending[cid] = (target, rec)
                if sk is not None:
                    sketches[cid] = sk
                supports[cid] = c["support"]
                errors.append(error)
                energies.append(energy)
                t_encode += time.perf_counter() - t
            active = sorted(payloads)
            if not active:
                for cid, (target, _) in pending.items():
                    if protocol.error_feedback:
                        state["memories"][cid].rollback(target)
                state["logs"].append(
                    dict(
                        round=r,
                        participants=0,
                        status="no_feasible_clients",
                        uplink_bytes=packetbytes,
                        downlink_bytes=0,
                        cumulative_uplink=state["uplink"],
                        cumulative_downlink=state["downlink"],
                    )
                )
                state["momentum"] = np.zeros(model.size)
                state["round"] = r
                if checkpoint:
                    atomic_pickle(checkpoint, checkpoint_snapshot(state))
                continue
            t = time.perf_counter()
            excluded = {}
            audit_resid = audit_tol = np.nan
            if protocol.audit:
                excluded, residuals, audit_tol = audit_clients(sketcher, payloads, sketches, cfg.clip_norm)
                audit_resid = float(np.max(list(residuals.values()))) if residuals else np.nan
            t_audit = time.perf_counter() - t
            kept = [k for k in active if k not in excluded]
            if not kept:
                for cid, (target, _) in pending.items():
                    if protocol.error_feedback:
                        state["memories"][cid].rollback(target)
                state["momentum"] = np.zeros(model.size)
                for cid in active:
                    state["events"].append(
                        _event(
                            r,
                            cid,
                            clients,
                            base_caps,
                            budgets,
                            scenario,
                            audit_round_failed=True,
                            excluded=excluded.get(cid, "all_excluded"),
                            screened=False,
                            flagged=False,
                            weight=0.0,
                        )
                    )
                state["logs"].append(
                    dict(
                        round=r,
                        participants=len(active),
                        status="all_audit_excluded",
                        uplink_bytes=packetbytes,
                        sketch_bytes=sketchbytes,
                        downlink_bytes=0,
                        cumulative_uplink=state["uplink"] + packetbytes,
                        cumulative_downlink=state["downlink"],
                        audit_ok=False,
                        audit_residual=audit_resid,
                        audit_tolerance=audit_tol,
                        audit_excluded=len(excluded),
                    )
                )
                state["uplink"] += packetbytes
                state["round"] = r
                if checkpoint:
                    atomic_pickle(checkpoint, checkpoint_snapshot(state))
                continue
            t = time.perf_counter()
            if protocol.screen:
                trust, diag = screen({k: sketches[k] for k in kept}, cfg, r)
            else:
                trust = {k: 1.0 for k in kept}
                diag = {k: dict(screened=False, flagged=False, weight=1.0, reason="no_defense") for k in kept}
            t_screen = time.perf_counter() - t
            low_consensus = sum(1 for k in kept if diag[k].get("reason") in ("no_consensus", "median_not_converged"))
            weight = {k: trust[k] for k in kept}
            for cid in active:
                state["events"].append(
                    _event(
                        r,
                        cid,
                        clients,
                        base_caps,
                        budgets,
                        scenario,
                        audit_round_failed=False,
                        excluded=excluded.get(cid, ""),
                        screened=diag.get(cid, {}).get("screened", False),
                        flagged=diag.get(cid, {}).get("flagged", False),
                        weight=trust.get(cid, 0.0),
                        screen_reason=diag.get(cid, {}).get("reason", ""),
                    )
                )
            t = time.perf_counter()
            status = "ok" if not excluded else "audit_excluded_clients"
            delta = aggregate_updates(
                [payloads[k] for k in kept], [weight[k] for k in kept], [supports[k] for k in kept]
            )
            state["momentum"] = cfg.server_momentum * state["momentum"] + delta
            newtheta = theta + cfg.server_lr * state["momentum"]
            if not np.isfinite(newtheta).all():
                raise FloatingPointError("Nonfinite server model")
            state["theta"] = newtheta
            t_aggregate = time.perf_counter() - t
            if protocol.error_feedback:
                for cid, (target, rec) in pending.items():
                    if cid in kept:
                        state["memories"][cid].acknowledge(target, rec)
                    else:
                        state["memories"][cid].rollback(target)
            down = dense_size * len(kept)
            state["uplink"] += packetbytes
            state["downlink"] += down
            state["train_seconds"] += t_train
            state["encode_seconds"] += t_encode
            state["screen_seconds"] += t_screen
            state["audit_seconds"] += t_audit
            state["aggregate_seconds"] += t_aggregate
            val_score = val_nll = np.nan
            if r % cfg.eval_every == 0 or r == cfg.rounds:
                model.set_flat(state["theta"])
                vp = expit(model.logits(Xval))
                val_score = participant_score(val, vp)
                val_nll = participant_score(val, vp, metric="nll")
                if val_score > state["best_score"] + 1e-12:
                    state.update(
                        best_score=val_score,
                        best_theta=state["theta"].copy(),
                        best_round=r,
                        selected_uplink=state["uplink"],
                        selected_downlink=state["downlink"],
                    )
            state["logs"].append(
                dict(
                    round=r,
                    participants=len(kept),
                    status=status,
                    uplink_bytes=packetbytes,
                    sketch_bytes=sketchbytes,
                    downlink_bytes=down,
                    cumulative_uplink=state["uplink"],
                    cumulative_downlink=state["downlink"],
                    train_s=t_train,
                    encode_s=t_encode,
                    screen_s=t_screen,
                    audit_s=t_audit,
                    aggregate_s=t_aggregate,
                    round_s=time.perf_counter() - start,
                    local_steps=train_steps,
                    mean_squared_error=float(np.mean(errors)) if errors else np.nan,
                    relative_distortion=float(np.sum(errors) / max(np.sum(energies), 1e-12)) if errors else np.nan,
                    audit_ok=not excluded,
                    audit_residual=audit_resid,
                    audit_tolerance=audit_tol,
                    audit_excluded=len(excluded),
                    low_consensus_clients=low_consensus,
                    infeasible_clients=len(infeasible),
                    val_macro_f1=val_score,
                    val_nll=val_nll,
                    norm_delta=float(np.linalg.norm(state["theta"] - theta)),
                    realized_bad_fraction=float(np.mean([clients[k]["bad"] for k in kept])),
                    actual_bad_clients=sum(clients[k]["bad"] for k in kept),
                )
            )
            state["round"] = r
            if checkpoint and (r % cfg.eval_every == 0 or r == cfg.rounds):
                atomic_pickle(checkpoint, checkpoint_snapshot(state))
    if state["best_round"] == 0:
        state["best_theta"] = state["theta"].copy()
    model.set_flat(state["best_theta"])
    state["model_bytes_fp32"] = dense_size
    state["metadata"] = dict(
        raw_data_central_preprocessing=True,
        secure_aggregation_implemented=False,
        end_to_end_dp=False,
        application_bytes_only=True,
        uncapped_dense_transport=is_dense,
        downlink_mode="full_model_unicast_fp32",
        actual_malicious_clients=sum(c["bad"] for c in clients.values()),
        total_clients=len(clients),
        model_params=model.size,
        preprocessor_features=len(prep.features),
        calibration_access="separate subjects",
        cohort_training="independent per dataset",
        sampling="seeded, deterministic",
    )
    return model, state


def summarize_state(state):
    """Communication, screening and audit diagnostics of a finished run."""
    log = pd.DataFrame(state["logs"])
    ev = pd.DataFrame(state["events"])
    screened = ev[ev.screened] if len(ev) else ev

    def rate(mask):
        return float(screened.loc[mask, "flagged"].mean()) if len(screened) and mask.any() else np.nan

    bad = screened.malicious if len(screened) else pd.Series([], dtype=bool)
    nr = max(len(log), 1)
    cr = int(log.participants.sum()) if len(log) else 0

    def count(col, pred):
        return int(pred(log[col]).sum()) if col in log else 0

    return dict(
        uplink_bytes=state["uplink"],
        downlink_bytes=state["downlink"],
        total_bytes=state["uplink"] + state["downlink"],
        selected_uplink_bytes=state["selected_uplink"],
        selected_downlink_bytes=state["selected_downlink"],
        selected_round=state["best_round"],
        completed_rounds=state["round"],
        client_rounds=cr,
        screen_tpr=rate(bad),
        screen_fpr=rate(~bad),
        screened_malicious_events=int(bad.sum()),
        screened_honest_events=int((~bad).sum()),
        screening_coverage=float(ev.screened.mean()) if len(ev) else np.nan,
        mean_effective_malicious_weight=(
            float(ev[ev.malicious].weight.mean()) if len(ev) and ev.malicious.any() else np.nan
        ),
        audit_excluded_client_rounds=int(log.audit_excluded.sum()) if "audit_excluded" in log else 0,
        all_audit_excluded_rounds=count("status", lambda s: s == "all_audit_excluded"),
        low_consensus_client_rounds=int(log.low_consensus_clients.sum()) if "low_consensus_clients" in log else 0,
        infeasible_client_rounds=int(log.infeasible_clients.sum()) if "infeasible_clients" in log else 0,
        client_train_ms=state["train_seconds"] * 1000 / max(cr, 1),
        client_encode_ms=state["encode_seconds"] * 1000 / max(cr, 1),
        server_screen_ms=state["screen_seconds"] * 1000 / nr,
        server_audit_ms=state["audit_seconds"] * 1000 / nr,
        server_aggregate_ms=state["aggregate_seconds"] * 1000 / nr,
        model_params=state["metadata"]["model_params"],
        model_packet_fp32_bytes=state["model_bytes_fp32"],
        relative_distortion=float(log.relative_distortion.mean()) if "relative_distortion" in log else np.nan,
        metadata=state["metadata"],
    )
