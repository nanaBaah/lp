"""Assertion helpers shared by the test modules in this folder.

Kept out of ``conftest.py`` on purpose: pytest imports ``conftest`` itself,
so importing it by name from a test module risks loading it twice under two
module identities. Fixtures belong there, plain helpers belong here.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import _pad_or_trim

DT = 0.25
ATOL = 1e-6


def check_invariants(result, cfg, prices, soc0, *, avail_gen=None, avail_cons=None):
    """Assert the properties every optimal solution must have.

    Deliberately re-derives each quantity from the returned dispatch rather
    than trusting the solver's own bookkeeping. Shared across variants: any
    ``solve(cfg, prices, soc0, **kwargs) -> dict`` result must satisfy this,
    which is what makes it usable from both ``tests/test_variants.py`` and
    ``tests/test_lp.py``.
    """
    assert result["status"] == "optimal"
    T = len(prices)
    gen, cons, soc = result["gen"], result["cons"], result["soc"]

    assert gen.shape == cons.shape == soc.shape == (T,)

    assert (gen >= -ATOL).all(), "generation must be non-negative"
    assert (cons >= -ATOL).all(), "consumption must be non-negative"

    # Reservoir balance: SoC_t = SoC_{t-1} − gen·Δt/η_gen + cons·Δt·η_cons
    prev = np.concatenate([[soc0], soc[:-1]])
    expected_soc = (
        prev
        - gen * DT / cfg["eta_gen"]
        + cons * DT * cfg["eta_cons"]
    )
    np.testing.assert_allclose(soc, expected_soc, atol=1e-6)

    assert soc.min() >= cfg["SoC_min_mwh"] - ATOL
    assert soc.max() <= cfg["SoC_max_mwh"] + ATOL

    net = gen - cons
    np.testing.assert_allclose(result["net"], net, atol=ATOL)
    rev = prices * net * DT
    np.testing.assert_allclose(result["revenue_per_step"], rev, atol=1e-6)
    assert result["revenue_total"] == pytest.approx(rev.sum(), abs=1e-6)

    # Mutual exclusion, with the same denominators the solver uses
    ag = (
        np.full(T, float(cfg["P_gen_max_mw"]))
        if avail_gen is None
        else _pad_or_trim(np.asarray(avail_gen, dtype=float), T)
    )
    ap = (
        np.full(T, float(cfg["P_cons_max_mw"]))
        if avail_cons is None
        else _pad_or_trim(np.asarray(avail_cons, dtype=float), T)
    )
    frac = gen / np.maximum(ag, 1.0) + cons / np.maximum(ap, 1.0)
    assert frac.max() <= 1 + 1e-6
