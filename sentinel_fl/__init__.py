"""SENTINEL-FL: byte-capped, screened and audited federated learning for wearable sensing."""

from .config import PRESETS, Config, __version__, data_config, get_config
from .federated import LR_SEARCH, SENTINEL_FL, Protocol, Scenario, run_federated
from .inference import SentinelFLPredictor
from .pipeline import run_fold, run_loso

__all__ = [
    "Config",
    "PRESETS",
    "get_config",
    "data_config",
    "Protocol",
    "Scenario",
    "SENTINEL_FL",
    "LR_SEARCH",
    "run_federated",
    "run_fold",
    "run_loso",
    "SentinelFLPredictor",
    "__version__",
]
