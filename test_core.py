"""Tests for variant-agnostic building blocks: FLEET, curtailment/redispatch
window helpers, and the synthetic price generator. Nothing here calls a
solver — see ``test_variants.py`` (shared across variants) and ``test_lp.py``
(LP-only mechanics) for that.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import (
    AID_NAME,
    FLEET,
    NAME_AID,
    _pad_or_trim,
    generate_price_curves,
    make_curtailment_array,
    make_redispatch_constraints,
)


@pytest.mark.parametrize("aid", sorted(FLEET))
def test_fleet_entry_is_a_valid_solver_config(aid):
    cfg = FLEET[aid]
    required = [
        "P_gen_max_mw", "P_cons_max_mw", "E_reservoir_mwh",
        "eta_gen", "eta_cons", "SoC_min_mwh", "SoC_max_mwh",
    ]
    for key in required:
        assert key in cfg, f"{cfg['name']} missing {key}"
        assert float(cfg[key]) >= 0

    assert cfg["SoC_min_mwh"] < cfg["SoC_max_mwh"]
    assert cfg["SoC_max_mwh"] <= cfg["E_reservoir_mwh"]


@pytest.mark.parametrize("aid", sorted(FLEET))
def test_efficiency_convention_holds(aid):
    """Round-trip efficiency sits entirely on the consumption side."""
    cfg = FLEET[aid]
    assert cfg["eta_gen"] == 1.0
    assert 0 < cfg["eta_cons"] < 1


@pytest.mark.parametrize("aid", sorted(FLEET))
def test_machine_counts_match_btag_lists(aid):
    cfg = FLEET[aid]
    assert len(cfg["turbine_btags"]) == cfg["n_turbines"]
    assert len(cfg["pump_btags"]) == cfg["n_pumps"]


def test_name_lookups_round_trip():
    assert NAME_AID == {v: k for k, v in AID_NAME.items()}
    assert set(AID_NAME) == set(FLEET)


def test_pad_or_trim_trims_long_arrays():
    np.testing.assert_array_equal(_pad_or_trim(np.arange(10.0), 4), [0, 1, 2, 3])


def test_pad_or_trim_repeats_the_last_value():
    np.testing.assert_array_equal(_pad_or_trim(np.array([1.0, 2.0]), 5), [1, 2, 2, 2, 2])


def test_curtailment_array_is_a_fraction_of_availability():
    avail = np.array([800.0, 800.0, 600.0, 600.0])
    np.testing.assert_allclose(
        make_curtailment_array(avail, 0.10), [80, 80, 60, 60]
    )


def test_curtailment_array_is_never_negative():
    avail = np.array([-50.0, 0.0, 100.0])
    assert (make_curtailment_array(avail, 0.5) >= 0).all()


def test_redispatch_constraints_cover_the_half_open_window():
    rc = make_redispatch_constraints(72, 80, gen_max=0)
    assert sorted(rc) == list(range(72, 80))
    assert rc[72] == {"gen_max": 0.0}
    assert "cons_max" not in rc[72]


def test_redispatch_constraints_include_consumption_cap_when_given():
    rc = make_redispatch_constraints(0, 2, gen_max=10, cons_max=20)
    assert rc[0] == {"gen_max": 10.0, "cons_max": 20.0}


def test_price_generator_is_deterministic():
    first_da, first_real = generate_price_curves()
    second_da, second_real = generate_price_curves()
    assert first_da.shape == first_real.shape == (192,)
    np.testing.assert_array_equal(first_da, second_da)
    np.testing.assert_array_equal(first_real, second_real)
