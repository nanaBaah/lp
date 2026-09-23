"""Shared pytest setup for this folder.

pytest finds this file on its own — you never import it or run it.

Run the suite with::

    .venv/bin/python -m pytest solutions/innovations/2026_new_redispatch_mechanism -q

or, from the Databricks UI, open ``run_tests.py`` and run all cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from core import FLEET, generate_price_curves

GOLD = 133


@pytest.fixture(name="da_prices", scope="session")
def fixture_da_prices() -> np.ndarray:
    da, _ = generate_price_curves()
    return da


@pytest.fixture(name="gold")
def fixture_gold() -> dict:
    return FLEET[GOLD].copy()
