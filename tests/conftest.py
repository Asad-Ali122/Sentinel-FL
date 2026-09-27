import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel_fl.config import PRESETS, data_config  # noqa: E402
from sentinel_fl.features import load_data  # noqa: E402


@pytest.fixture(scope="session")
def fixture_frame():
    df, _ = load_data(data_config(), synthetic=True)
    return df


@pytest.fixture()
def smoke_cfg():
    return PRESETS["smoke"]
