"""Invariant tests that apply to every redispatch variant.

Every test takes the ``solve`` fixture rather than calling a variant
directly, so adding an entry to VARIANTS below runs this whole suite
against the new variant without writing new tests. A failure here is a
solver bug, not a known limitation.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import FLEET, make_curtailment_array, make_redispatch_constraints
from helpers import ATOL, check_invariants
from variants.lp import solve_lp

VARIANTS = {"lp": solve_lp}


@pytest.fixture(params=sorted(VARIANTS))
def solve(request):
    return VARIANTS[request.param]


# ═══════════════════════════════════════════════════════════════════════════
# Core invariants
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("aid", sorted(FLEET))
def test_unconstrained_solve_satisfies_invariants(solve, aid, da_prices):
    cfg = FLEET[aid].copy()
    soc0 = cfg["E_reservoir_mwh"] / 2
    result = solve(cfg, da_prices, soc0)
    check_invariants(result, cfg, da_prices, soc0)


def test_invariants_hold_with_every_restriction_engaged(solve, gold, da_prices):
    T = len(da_prices)
    avail_gen = np.full(T, 900.0)
    avail_cons = np.full(T, 1000.0)
    reserves = np.full(T, 150.0)
    soc0 = 5000.0

    result = solve(
        gold, da_prices, soc0,
        avail_gen=avail_gen,
        avail_cons=avail_cons,
        curtail_gen=make_curtailment_array(avail_gen, 0.2),
        curtail_cons=make_curtailment_array(avail_cons, 0.1),
        redispatch_constraints=make_redispatch_constraints(72, 80, gen_max=0),
        enforce_terminal_soc=True,
        reserve_schedule_gen=reserves,
        reserve_schedule_cons=reserves,
        min_stable_gen_mw=50.0,
        min_stable_cons_mw=50.0,
    )
    # Curtailment is applied before the mutual-exclusion denominators are formed
    check_invariants(
        result, gold, da_prices, soc0,
        avail_gen=avail_gen * 0.8,
        avail_cons=avail_cons * 0.9,
    )


def test_revenue_eval_is_the_prefix_of_revenue_total(solve, gold, da_prices):
    result = solve(gold, da_prices, 5000, eval_qh=96)
    assert result["revenue_eval"] == pytest.approx(
        result["revenue_per_step"][:96].sum(), abs=1e-6
    )


def test_soc_never_leaves_band_from_extreme_starts(solve, gold, da_prices):
    for soc0 in (gold["SoC_min_mwh"], gold["SoC_max_mwh"]):
        result = solve(gold, da_prices, soc0)
        check_invariants(result, gold, da_prices, soc0)


def test_plant_does_not_generate_and_consume_in_the_same_quarter_hour(solve, gold, da_prices):
    """No variant may run both directions at once on a realistic price curve.

    Worth asserting separately from ``check_invariants``, which only bounds
    the mutual-exclusion fraction at 1 and so still passes when both
    directions run at part load. A variant that relaxes the constraint
    instead of enforcing it — the LP expresses "one direction at a time" as
    a fraction rather than a binary, to stay an LP — can leak here.
    """
    result = solve(gold, da_prices, 5000)
    both = (result["gen"] > 1e-6) & (result["cons"] > 1e-6)
    assert not both.any()


# ═══════════════════════════════════════════════════════════════════════════
# Bound cascade: availability → curtailment → reserves → redispatch
# ═══════════════════════════════════════════════════════════════════════════

def test_generation_respects_availability(solve, gold, da_prices):
    avail_gen = np.full(len(da_prices), 400.0)
    result = solve(gold, da_prices, 5000, avail_gen=avail_gen)
    assert result["gen"].max() <= 400.0 + ATOL


def test_reserves_are_deducted_from_headroom(solve, gold, da_prices):
    T = len(da_prices)
    reserved = 200.0
    result = solve(
        gold, da_prices, 5000,
        avail_gen=np.full(T, 1000.0),
        reserve_schedule_gen=np.full(T, reserved),
    )
    assert result["gen"].max() <= 1000.0 - reserved + ATOL


def test_holding_reserves_cannot_increase_energy_revenue(solve, gold, da_prices):
    T = len(da_prices)
    avail_gen = np.full(T, 1000.0)
    free = solve(gold, da_prices, 5000, avail_gen=avail_gen)
    held = solve(
        gold, da_prices, 5000,
        avail_gen=avail_gen,
        reserve_schedule_gen=np.full(T, 200.0),
    )
    assert held["revenue_total"] <= free["revenue_total"] + ATOL


def test_per_qh_reserve_schedule_overrides_scalar_config(solve, gold, da_prices):
    T = len(da_prices)
    gold["P_reserved_gen_mw"] = 900.0
    result = solve(
        gold, da_prices, 5000,
        avail_gen=np.full(T, 1000.0),
        reserve_schedule_gen=np.zeros(T),
    )
    assert result["gen"].max() > 100.0, "scalar reserve should have been overridden"


def test_redispatch_gen_max_zero_shuts_generation_off_in_window(solve, gold, da_prices):
    result = solve(
        gold, da_prices, 5000,
        redispatch_constraints=make_redispatch_constraints(72, 80, gen_max=0),
    )
    assert result["gen"][72:80].max() <= ATOL
    assert result["gen"][:72].max() > ATOL, "only the window should be cut"


def test_feasible_redispatch_floor_is_honoured(solve, gold, da_prices):
    """A forced-operation instruction the plant can actually meet."""
    rc = {t: {"gen_min": 200.0} for t in range(8)}
    result = solve(gold, da_prices, 5000, redispatch_constraints=rc)
    assert result["gen"][:8].min() >= 200.0 - ATOL


def test_min_stable_output_applies_only_where_reserves_are_held(solve, gold, da_prices):
    T = len(da_prices)
    reserves = np.zeros(T)
    reserves[:8] = 50.0
    result = solve(
        gold, da_prices, 5000,
        reserve_schedule_gen=reserves,
        min_stable_gen_mw=100.0,
    )
    assert result["gen"][:8].min() >= 100.0 - ATOL
    assert result["gen"][8:12].min() <= ATOL, "no floor where no reserves are held"


# ═══════════════════════════════════════════════════════════════════════════
# Opportunity cost: the paired-delta the whole method rests on
# ═══════════════════════════════════════════════════════════════════════════

def test_curtailment_never_increases_revenue(solve, gold, da_prices):
    T = len(da_prices)
    avail_gen = np.full(T, gold["P_gen_max_mw"])
    previous = None
    for fraction in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
        result = solve(
            gold, da_prices, 5000,
            avail_gen=avail_gen,
            curtail_gen=make_curtailment_array(avail_gen, fraction),
        )
        assert result["status"] == "optimal"
        if previous is not None:
            assert result["revenue_total"] <= previous + ATOL
        previous = result["revenue_total"]


def test_opportunity_cost_is_non_negative(solve, gold, da_prices):
    T = len(da_prices)
    avail_gen = np.full(T, gold["P_gen_max_mw"])
    free = solve(gold, da_prices, 5000, avail_gen=avail_gen)
    curtailed = solve(
        gold, da_prices, 5000,
        avail_gen=avail_gen,
        curtail_gen=make_curtailment_array(avail_gen, 0.3),
    )
    assert free["revenue_total"] - curtailed["revenue_total"] >= -ATOL


def test_zero_curtailment_is_identical_to_no_curtailment(solve, gold, da_prices):
    T = len(da_prices)
    avail_gen = np.full(T, 800.0)
    plain = solve(gold, da_prices, 5000, avail_gen=avail_gen)
    zeroed = solve(
        gold, da_prices, 5000,
        avail_gen=avail_gen,
        curtail_gen=np.zeros(T),
    )
    assert zeroed["revenue_total"] == pytest.approx(plain["revenue_total"], abs=1e-6)


def test_full_curtailment_leaves_no_generation(solve, gold, da_prices):
    T = len(da_prices)
    avail_gen = np.full(T, gold["P_gen_max_mw"])
    result = solve(
        gold, da_prices, 5000,
        avail_gen=avail_gen,
        curtail_gen=make_curtailment_array(avail_gen, 1.0),
    )
    assert result["gen"].max() <= ATOL


def test_better_round_trip_efficiency_is_never_worse(solve, gold, da_prices):
    """Gaming resistance: understating η_cons cannot raise a variant's revenue.

    The opportunity cost is a difference of two runs, so a plant that
    misreports efficiency to inflate its claim only lowers both sides.
    """
    previous = None
    for eta in (0.5, 0.6, 0.7, 0.782, 0.9, 1.0):
        cfg = gold.copy()
        cfg["eta_cons"] = eta
        result = solve(cfg, da_prices, 5000)
        if previous is not None:
            assert result["revenue_total"] >= previous - ATOL
        previous = result["revenue_total"]


# ═══════════════════════════════════════════════════════════════════════════
# Terminal SoC
# ═══════════════════════════════════════════════════════════════════════════

def test_terminal_soc_returns_to_start_when_enforced(solve, gold, da_prices):
    soc0 = 5000.0
    tol = 0.01 * (gold["SoC_max_mwh"] - gold["SoC_min_mwh"])
    result = solve(gold, da_prices, soc0, enforce_terminal_soc=True)
    assert abs(result["soc"][-1] - soc0) <= tol + ATOL


def test_unenforced_terminal_soc_drains_the_reservoir(solve, gold, da_prices):
    """Without the circular boundary a variant sells stored water and keeps it.

    This is why the validation notebook passes enforce_terminal_soc=True for
    free-vs-curtailed comparisons.
    """
    soc0 = 5000.0
    free = solve(gold, da_prices, soc0)
    circular = solve(gold, da_prices, soc0, enforce_terminal_soc=True)
    assert free["soc"][-1] < circular["soc"][-1]
    assert free["revenue_total"] > circular["revenue_total"]


def test_terminal_soc_tolerance_is_respected(solve, gold, da_prices):
    soc0 = 5000.0
    result = solve(
        gold, da_prices, soc0,
        enforce_terminal_soc=True,
        terminal_soc_tol_frac=0.0,
    )
    assert result["soc"][-1] == pytest.approx(soc0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Infeasibility
# ═══════════════════════════════════════════════════════════════════════════

def test_impossible_instruction_reports_infeasible_without_dispatch_keys(solve, gold, da_prices):
    """Forcing full output for a week drains the reservoir past its floor."""
    rc = {t: {"gen_min": 1060.0, "gen_max": 1060.0} for t in range(len(da_prices))}
    result = solve(gold, da_prices, 5000, redispatch_constraints=rc)
    assert result["status"] == "infeasible"
    assert "message" in result
    assert "gen" not in result


def test_reserve_headroom_conflict_is_reported_distinctly(solve, gold, da_prices):
    """Reserves needing more SoC headroom than the band has.

    Callers need to tell this apart from ordinary infeasibility, so a
    variant names it and reports what the relaxed run would have earned.
    """
    cfg = gold.copy()
    cfg["SoC_min_mwh"] = 400
    cfg["SoC_max_mwh"] = 600
    result = solve(
        cfg, da_prices, 500,
        reserve_schedule_gen=np.full(len(da_prices), 1060.0),
    )
    assert result["status"] == "infeasible_fortsetzungsfehler"
    assert "relaxed_revenue_total" in result
    assert "relaxed_revenue_eval" in result


# ═══════════════════════════════════════════════════════════════════════════
# Horizon handling
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("T", [1, 4, 96, 192, 672])
def test_horizon_length_is_taken_from_the_price_array(solve, gold, T):
    prices = np.linspace(20, 80, T)
    result = solve(gold, prices, 5000)
    check_invariants(result, gold, prices, 5000)


def test_eval_window_longer_than_horizon_falls_back_to_total(solve, gold):
    prices = np.linspace(20, 80, 96)
    result = solve(gold, prices, 5000, eval_qh=999)
    assert result["revenue_eval"] == pytest.approx(result["revenue_total"], abs=1e-6)
