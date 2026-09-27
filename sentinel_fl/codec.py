"""Serialized byte-capped update codec, error feedback and fleet byte allocation."""

from __future__ import annotations

import math
import struct

import numpy as np

PACKET_HEADER = struct.Struct("<4sIHH")
RECORD_HEADER = struct.Struct("<HBBIf")
ENVELOPE_HEADER = struct.Struct("<4sIIII")  # magic, round, client slot, payload length, sketch length
TELEMETRY = struct.Struct("<ff")  # squared error and target energy


class InfeasibleBudget(ValueError):
    """Raised when a byte cap cannot carry even one coordinate."""


def pack_unsigned(q, bits):
    """Bit-pack non-negative integers of width ``bits``."""
    q = np.asarray(q, np.uint32)
    if bits == 8:
        return q.astype("u1").tobytes()
    if bits == 16:
        return q.astype("<u2").tobytes()
    mask = (q[:, None] >> np.arange(bits) & 1).astype(np.uint8)
    return np.packbits(mask.ravel(), bitorder="little").tobytes()


def unpack_unsigned(data, bits, n):
    """Inverse of :func:`pack_unsigned`."""
    if bits == 8:
        return np.frombuffer(data, dtype="u1", count=n).astype(np.int32)
    if bits == 16:
        return np.frombuffer(data, dtype="<u2", count=n).astype(np.int32)
    a = np.unpackbits(np.frombuffer(data, dtype="u1"), bitorder="little")[: n * bits].reshape(n, bits)
    return (a * (1 << np.arange(bits))).sum(axis=1).astype(np.int32)


def index_mode(n, k):
    """Cheapest position encoding for ``k`` of ``n`` coordinates.

    Returns ``(mode, bytes)``: 0 dense, 1/2 explicit 16/32-bit indices, 3 bitmap."""
    if k == n:
        return (0, 0)
    width = 2 if n <= 65536 else 4
    if math.ceil(n / 8) < k * width:
        return (3, math.ceil(n / 8))
    return (1 if width == 2 else 2, k * width)


def record_cost(n, k, bits):
    """Exact serialized size in bytes of one record."""
    return RECORD_HEADER.size + index_mode(n, k)[1] + math.ceil(k * bits / 8)


QUANT_SCALE_CANDIDATES = (1.0, 0.85, 0.7, 0.55, 0.4)


def quantize_values(vals, bits):
    """Uniform quantiser over the full ``2**bits`` alphabet with an MSE-optimal scale.

    The scale is chosen among a few fractions of the peak magnitude so that one outlier does not
    set the step for the whole record. Returns ``(scale, integer_codes, reconstruction)``."""
    k = len(vals)
    if k == 0:
        return (0.0, np.zeros(0, int), np.zeros(0))
    peak = float(np.max(np.abs(vals)))
    if not np.isfinite(peak):
        raise ValueError("Nonfinite quantizer scale")
    if peak == 0.0:
        return (0.0, np.zeros(k, int), np.zeros(k))
    L = 1 << bits - 1
    best = None
    for frac in QUANT_SCALE_CANDIDATES:
        scale = float(np.float32(peak * frac))
        if scale <= 0:
            continue
        q = np.clip(np.rint(vals / scale * L), -L, L - 1).astype(int)
        rec = q.astype(float) / L * scale
        err = float(np.sum((vals - rec) ** 2))
        if best is None or err < best[0] - 1e-18:
            best = (err, scale, q, rec)
    return (best[1], best[2], best[3])


def make_record(v, block_id, bits, idx=None):
    """Serialize one block (or the whole vector) with a chosen bit depth and coordinate subset."""
    v = np.asarray(v, float)
    n = len(v)
    idx = np.arange(n) if idx is None else np.sort(np.asarray(idx, int))
    k = len(idx)
    mode, _ = index_mode(n, k)
    if mode == 0:
        pos = b""
    elif mode in (1, 2):
        pos = idx.astype("<u2" if mode == 1 else "<u4").tobytes()
    else:
        mask = np.zeros(n, np.uint8)
        mask[idx] = 1
        pos = np.packbits(mask, bitorder="little").tobytes()
    vals = v[idx]
    if bits == 32:
        scale = 0.0
        body = vals.astype("<f4").tobytes()
        qrec = np.frombuffer(body, dtype="<f4").astype(float)
    else:
        L = 1 << bits - 1
        scale, q, qrec = quantize_values(vals, bits)
        body = pack_unsigned(np.asarray(q, int) + L, bits)
    rec = np.zeros(n)
    rec[idx] = qrec
    packet = RECORD_HEADER.pack(block_id, bits, mode, k, scale) + pos + body
    if len(packet) != record_cost(n, k, bits):
        raise AssertionError("Record accounting")
    return (packet, rec)


def make_packet(dim, records):
    """Wrap serialized records in a packet header."""
    return PACKET_HEADER.pack(b"SNTL", int(dim), len(records), 0) + b"".join(records)


class ByteCodec:
    """Serialized, byte-capped update codec.

    A packet is a header plus records. The encoder builds several candidates (block-wise rate
    allocation, top-magnitude sparse packet, dense packets at each bit depth), and sends the one
    with the smallest squared error that fits the hard byte cap. The server only ever decodes
    the bytes it receives."""

    def __init__(self, blocks, dim, keep=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0), bits=(2, 4, 8, 16, 32)):
        self.blocks = list(blocks)
        self.dim = int(dim)
        self.keep = tuple(keep)
        self.bits = tuple(bits)

    def decode(self, packet):
        """Strictly validate and decode a packet; raises ``ValueError`` on any malformed input."""
        if len(packet) < PACKET_HEADER.size:
            raise ValueError("Truncated packet")
        magic, dim, nrec, flags = PACKET_HEADER.unpack_from(packet)
        if magic != b"SNTL" or dim != self.dim or flags != 0:
            raise ValueError("Wrong packet schema")
        pos = PACKET_HEADER.size
        out = np.zeros(dim)
        seen = set()
        for _ in range(nrec):
            if pos + RECORD_HEADER.size > len(packet):
                raise ValueError("Truncated record")
            bid, bits, mode, k, scale = RECORD_HEADER.unpack_from(packet, pos)
            pos += RECORD_HEADER.size
            if bits not in (2, 4, 8, 16, 32) or mode not in (0, 1, 2, 3):
                raise ValueError("Invalid code")
            if bid in seen or 65535 in seen or (bid == 65535 and nrec != 1):
                raise ValueError("Overlapping blocks")
            seen.add(bid)
            if bid == 65535:
                lo, hi = (0, dim)
            elif bid < len(self.blocks):
                _, lo, hi = self.blocks[bid]
            else:
                raise ValueError("Unknown block")
            n = hi - lo
            if k > n or not np.isfinite(scale) or scale < 0:
                raise ValueError("Invalid dimensions/scale")
            if mode == 0:
                if k != n:
                    raise ValueError("Invalid dense count")
                idx = np.arange(n)
            elif mode in (1, 2):
                width = 2 if mode == 1 else 4
                nb = k * width
                if pos + nb > len(packet):
                    raise ValueError("Truncated positions")
                idx = np.frombuffer(packet, dtype="<u2" if mode == 1 else "<u4", count=k, offset=pos).astype(int)
                pos += nb
                if len(np.unique(idx)) != k or (k and idx.max() >= n):
                    raise ValueError("Invalid positions")
            else:
                nb = math.ceil(n / 8)
                if pos + nb > len(packet):
                    raise ValueError("Truncated bitmap")
                mask = np.unpackbits(np.frombuffer(packet[pos : pos + nb], dtype="u1"), bitorder="little")[:n]
                pos += nb
                idx = np.flatnonzero(mask)
                if len(idx) != k:
                    raise ValueError("Bitmap count mismatch")
            nb = math.ceil(k * bits / 8)
            if pos + nb > len(packet):
                raise ValueError("Truncated values")
            body = packet[pos : pos + nb]
            pos += nb
            if bits == 32:
                vals = np.frombuffer(body, dtype="<f4").astype(float)
            else:
                L = 1 << bits - 1
                q = unpack_unsigned(body, bits, k)
                if np.any(q > 2 * L - 1):
                    raise ValueError("Reserved quantization code")
                vals = (q - L) / L * scale
            out[lo + idx] = vals
        if pos != len(packet) or not np.isfinite(out).all():
            raise ValueError("Trailing/nonfinite payload")
        return out

    def dense(self, v):
        """Full-precision (fp32) packet of the whole vector."""
        p, rec = make_record(v, 65535, 32)
        return (make_packet(self.dim, [p]), rec)

    def sparse_packet(self, v, budget):
        """Best (k, bit-depth) top-magnitude packet that fits the budget."""
        n = len(v)
        room = int(budget) - PACKET_HEADER.size
        best = None
        for b in self.bits:
            if room < record_cost(n, 1, b):
                continue
            if record_cost(n, n, b) <= room:
                k = n
            else:
                lo, hi = (1, n - 1)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if record_cost(n, mid, b) <= room:
                        lo = mid
                    else:
                        hi = mid - 1
                k = lo
            idx = np.argsort(-np.abs(v), kind="stable")[:k]
            p, rec = make_record(v, 65535, b, idx)
            err = float(np.sum((v - rec) ** 2))
            if best is None or (err, len(p)) < (best[0], best[1]):
                best = (err, len(p), p, rec)
        if best is None:
            raise InfeasibleBudget(f"budget {budget} cannot carry one coordinate of {n}")
        return (make_packet(self.dim, [best[2]]), best[3])

    def block_packet(self, v, budget):
        """Greedy block-wise rate allocation.

        Every parameter block gets a Pareto menu of (bytes, distortion) options; options are upgraded
        by best marginal distortion reduction per byte while the budget allows."""
        menus = []
        for bid, (_, lo, hi) in enumerate(self.blocks):
            x = v[lo:hi]
            order = np.argsort(-np.abs(x), kind="stable")
            opts = [(0, float(x @ x), b"")]
            for k in sorted({max(1, int(round(f * len(x)))) for f in self.keep}):
                for bits in self.bits:
                    p, rec = make_record(x, bid, bits, order[:k])
                    err = float(np.sum((x - rec) ** 2))
                    opts.append((len(p), err, p))
            frontier = []
            best = np.inf
            for item in sorted(opts, key=lambda a: (a[0], a[1])):
                if item[1] < best - 1e-20:
                    frontier.append(item)
                    best = item[1]
            menus.append(frontier)
        cur = [0] * len(menus)
        used = PACKET_HEADER.size
        while True:
            best = None
            for b, menu in enumerate(menus):
                old = menu[cur[b]]
                for j in range(cur[b] + 1, len(menu)):
                    dc = menu[j][0] - old[0]
                    gain = old[1] - menu[j][1]
                    if dc > 0 and gain > 0 and (used + dc <= budget):
                        score = gain / dc
                        if best is None or score > best[0]:
                            best = (score, b, j, dc)
            if best is None:
                break
            _, b, j, dc = best
            used += dc
            cur[b] = j
        records = [menus[b][j][2] for b, j in enumerate(cur) if menus[b][j][2]]
        packet = make_packet(self.dim, records)
        return (packet, self.decode(packet))

    def encode(self, v, budget, mode="sentinel"):
        """Encode ``v`` into the smallest-error packet that fits ``budget`` bytes.

        ``mode="sentinel"`` searches block-wise allocation, a top-magnitude sparse packet and dense packets
        at every bit depth; ``mode="dense"`` sends the full fp32 vector (learning-rate search only).
        Returns ``dict(packet, recon, error, bytes)``; raises :class:`InfeasibleBudget` if nothing fits.
        """
        v = np.asarray(v, float)
        budget = int(budget)
        if budget < PACKET_HEADER.size:
            raise ValueError("Budget cannot contain the empty packet header")
        candidates = []
        if mode == "dense":
            candidates = [self.dense(v)]
        elif mode == "sentinel":
            candidates = [self.block_packet(v, budget)]
            try:
                candidates.append(self.sparse_packet(v, budget))
            except InfeasibleBudget:
                pass
            for bits in self.bits:
                if PACKET_HEADER.size + record_cost(len(v), len(v), bits) <= budget:
                    p, rec = make_record(v, 65535, bits)
                    candidates.append((make_packet(self.dim, [p]), rec))
        else:
            raise ValueError(mode)
        options = []
        for p, rec in candidates:
            if len(p) <= budget:
                options.append((float(np.sum((v - rec) ** 2)), len(p), p, rec))
        if not options:
            raise InfeasibleBudget(f"{mode} has no feasible packet in {budget} bytes")
        err, _, p, rec = min(options, key=lambda a: (a[0], a[1]))
        if not np.allclose(rec, self.decode(p), rtol=0, atol=1e-12):
            raise AssertionError("Builder reconstruction disagrees with the decoder")
        return dict(packet=p, recon=rec, error=err, bytes=len(p))


class ErrorMemory:
    """Client-side error feedback with explicit rollback.

    ``target`` adds the residual to a fresh update, ``acknowledge`` stores what was not transmitted
    once the round was applied, and ``rollback`` keeps the entire target owed if it was not."""

    def __init__(self, dim):
        self.residual = np.zeros(dim)
        self._pending = None

    def target(self, delta, support=None):
        t = delta + self.residual
        if support is not None:
            t = np.where(np.asarray(support, bool), t, 0.0)
        self._pending = t
        return t

    def acknowledge(self, target, decoded):
        self.residual = target - decoded
        self._pending = None

    def rollback(self, target):
        self.residual = np.array(target, copy=True)
        self._pending = None


def bounded_allocate(caps, total, minimum, priorities=None):
    """Split a total byte budget across clients under per-client caps and a common floor."""
    ids = sorted(caps)
    cap = np.array([int(caps[k]) for k in ids])
    total = int(total)
    if total < 0 or (cap < 0).any():
        raise ValueError("Negative budget")
    total = min(total, int(cap.sum()))
    floor = np.array([min(int(minimum), c) for c in cap])
    if floor.sum() > total:
        raise ValueError("Fleet budget smaller than mandatory floor")
    x = floor.astype(float)
    remaining = total - x.sum()
    p = np.array([max(float((priorities or {}).get(k, 1.0)), 1e-12) for k in ids])
    while remaining > 1e-07:
        active = cap - x > 1e-07
        if not active.any():
            break
        grant = remaining * p[active] / p[active].sum()
        grant = np.minimum(grant, cap[active] - x[active])
        x[active] += grant
        remaining -= grant.sum()
    ints = np.floor(x + 1e-09).astype(int)
    left = total - int(ints.sum())
    for j in np.argsort(-(x - ints), kind="stable"):
        if left <= 0:
            break
        if ints[j] < cap[j]:
            ints[j] += 1
            left -= 1
    if ints.sum() != total or np.any(ints > cap):
        raise AssertionError("Allocation violated constraints")
    return {k: int(v) for k, v in zip(ids, ints)}
