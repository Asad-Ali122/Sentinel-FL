"""Load a trained SENTINEL-FL artifact and score new epochs/windows."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.special import expit


class SentinelFLPredictor:
    """Deployable bundle: preprocessor, residual model, calibration, sequence decoder and threshold.

    Artifacts are written to ``<out>/runs/<key>/artifact.pkl`` by a training run. They are Python
    pickles: load only artifacts that you created yourself or fully trust.

    ``predict_proba`` expects a feature frame with the columns produced by
    :func:`sentinel_fl.features.load_data` (including ``client_id`` and ``label_start``, which the
    sequence decoder uses to smooth each subject's own recording in time).
    """

    def __init__(self, bundle):
        self.bundle = bundle
        self.model = bundle["model"]
        self.preprocessor = bundle["preprocessor"]
        self.calibration = bundle["calibration"]
        self.decoder = bundle["decoder"]
        self.threshold = float(bundle["threshold"])
        self.sequence_decode = bool(bundle["sequence_decode"])
        self.dataset = bundle["dataset"]
        self.minority_class = bundle["minority_class"]

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as f:
            return cls(pickle.load(f))

    def predict_logits(self, df):
        """Raw (uncalibrated) model logits."""
        return self.model.logits(self.preprocessor.transform(df))

    def predict_proba(self, df):
        """Calibrated (and, if enabled, sequence-decoded) probability of class 1."""
        z = self.predict_logits(df)
        p = expit(self.calibration["a"] * np.clip(z, -30, 30) + self.calibration["b"])
        if self.sequence_decode:
            p = self.decoder.smooth(df, p)
        return p

    def predict(self, df):
        """Binary decisions at the threshold selected on validation subjects."""
        return (self.predict_proba(df) >= self.threshold).astype(int)
