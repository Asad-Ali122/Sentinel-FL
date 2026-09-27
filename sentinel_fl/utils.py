"""Deterministic RNG streams, hashing and atomic file helpers."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def rng_for(*parts):
    """Deterministic generator seeded by a hash of ``parts``; every random stream in the package is named this way."""
    h = hashlib.sha256(json.dumps(parts, default=str).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def digest(obj):
    """SHA-256 hex digest of a JSON-serialisable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def jsonable(x):
    """Recursively convert NumPy / Path objects to JSON types (non-finite floats become ``None``)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, obj):
    """Atomically write ``obj`` as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2, allow_nan=False))
    os.replace(tmp, path)


def atomic_pickle(path, obj):
    """Atomically pickle ``obj`` (write to a temporary file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=5)
    os.replace(tmp, path)


def hash_frame(df):
    """SHA-256 fingerprint of a DataFrame's columns, index and values."""
    h = hashlib.sha256("|".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()
