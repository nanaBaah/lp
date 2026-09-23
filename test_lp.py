"""LP-only mechanics: things that are specific to the linear-program
formulation rather than the shared ``solve()`` contract, plus its known
defects. Shared invariants for every variant live in ``test_variants.py``.

**Known defects** are marked ``xfail(strict=True)``. They assert the
behaviour we want from a solver that is currently wrong, so they report as
expected failures instead of red. When the solver is fixed they start
passing unexpectedly, which strict mode turns into a failure — forcing
whoever fixed it to delete the marker rather than leave a stale note behind.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import generate_price_curves, make_curtailment_array
from helpers import ATOL, DT
from variants.lp import solve_lp


def test_mutual_exclusion_caps_the_opposite_direction_when_it_binds(gold):
    """Forced generation at negative prices is the case where it bites.

    On ordinary price curves the constraint is economically redundant — the
    LP has no reason to run both ways at once, so the constraint could be
    deleted without changing any result. Here a forced-operation floor holds
    generation at half capacity while negative prices make the plant want to
    consume flat out, and the constraint is the only thing capping it.
    """
    T = 8
    gen_floor = gold["P_gen_max_mw"] / 2
    rc = {t: {"gen_min": gen_floor, "gen_max": gen_floor} for t in range(T)}
    result = solve_lp(gold, np.full(T, -50.0), 1000.0, redispatch_constraints=rc)

    assert result["status"] == "optimal"
    headroom = (1 - gen_floor / gold["P_gen_max_mw"]) * gold["P_cons_max_mw"]
    assert result["cons"].max() <= headroom + ATOL
    assert result["cons"].max() == pytest.approx(headroom, abs=1e-6), "cap should bind"


def test_consumption_fills_the_reservoir_at_eta_cons(gold):
    """Pins the efficiency convention inside the LP, not just in FLEET.

    Round-trip loss sits entirely on the consumption side, so 400 MW drawn
    for a quarter-hour must add 400 × 0.25 × η_cons MWh, and must add rather
    than remove it.
    """
    T = 4
    draw = 400.0
    rc = {t: {"gen_max": 0.0, "cons_min": draw, "cons_max": draw} for t in range(T)}
    result = solve_lp(gold, np.full(T, 10.0), 1000.0, redispatch_constraints=rc)

    assert result["status"] == "optimal"
    step = draw * DT * gold["eta_cons"]
    np.testing.assert_allclose(np.diff(result["soc"]), step, atol=1e-6)
    assert result["soc"][0] == pytest.approx(1000.0 + step, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Known defects
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.xfail(
    strict=True,
    reason="curtail_gen is edge-padded like avail_gen, so a cut shorter than "
           "the horizon silently extends to the end of it",
)
def test_short_curtailment_array_does_not_extend_past_its_length(gold):
    """A curtailment shorter than the horizon should stop where it ends.

    ``curtail_gen`` is padded by repeating its last value, the same as
    ``avail_gen``. Repeating the last *availability* value across an unknown
    future is reasonable. Repeating the last *curtailment* value is not: a
    caller passing a one-day cut into a seven-day run gets a plant that is
    curtailed all week, which inflates the opportunity cost several-fold
    with no warning. Zero-padding would fix it.
    """
    prices, _ = generate_price_curves()
    T = len(prices)
    avail_gen = np.full(T, gold["P_gen_max_mw"])
    day_one_cut = make_curtailment_array(np.full(96, gold["P_gen_max_mw"]), 1.0)

    result = solve_lp(gold, prices, 5000, avail_gen=avail_gen, curtail_gen=day_one_cut)

    assert result["gen"][96:].max() > ATOL, "day 2 was curtailed by a day-1 instruction"


@pytest.mark.xfail(
    strict=True,
    reason="the lb = min(lb, ub) feasibility guard lowers an unmeetable "
           "gen_min and still reports optimal",
)
def test_unmeetable_redispatch_floor_is_not_reported_as_optimal(gold, da_prices):
    """A forced-operation instruction the plant cannot meet should not pass silently.

    Availability caps generation at 300 MW while the instruction demands
    500 MW. Today the floor is quietly lowered to 300 and the run returns
    "optimal", so the caller cannot tell the instruction went unmet.

    How that should be signalled is left open — raising, or returning a
    distinct status, would both satisfy this test. What it rules out is
    claiming optimality over a dispatch that violates the instruction.
    """
    floor = 500.0
    rc = {t: {"gen_min": floor} for t in range(8)}

    try:
        result = solve_lp(
            gold, da_prices, 5000,
            avail_gen=np.full(len(da_prices), 300.0),
            redispatch_constraints=rc,
        )
    except ValueError:
        return  # rejecting the instruction outright is a valid fix

    if result["status"] == "optimal":
        assert result["gen"][:8].min() >= floor - ATOL, "floor was silently lowered"
