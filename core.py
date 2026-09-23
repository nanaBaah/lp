"""Shared building blocks for redispatch opportunity-cost variants.

When a grid operator curtails a storage plant, the loss is bigger than the
energy the plant could have sold during the curtailed hour. The plant also
loses the freedom to shift that energy into a better-priced hour later on.
This package estimates that full loss over time for any asset that stores
energy and gives it back — pumped hydro, a battery, anything with a
reservoir and a round-trip efficiency.

The method is to schedule the plant twice against the same prices, once
with the curtailment and once without, and take the revenue difference.
The scheduling itself is pluggable: each approach under ``variants/``
implements the same contract,

    solve(cfg, prices, soc0, **kwargs) -> dict

This module holds everything that contract does not depend on the solving
approach: fleet configuration, array/window helpers, and the synthetic
price generator used by the test suite. Algorithm-specific documentation
(e.g. the LP's objective and constraints) lives in the corresponding
``variants/<name>.py``.

Terminology
-----------
The plant has two directions, named for what the grid sees:

    generation  (``gen``)   — the plant feeds power into the grid
    consumption (``cons``)  — the plant draws power from the grid

On a pumped-storage plant these are turbine and pump mode; on a battery
they are discharging and charging. The grid-side pair is used everywhere
so a variant reads the same whichever technology it is pointed at, and so
it lines up with how redispatch is instructed (Erzeugung / Verbrauch).

Both are non-negative MW. Net power is ``gen − cons``, so net is negative
while the plant consumes. The ``*_btags`` and ``n_turbines`` / ``n_pumps``
entries in FLEET keep the hydro names, because they count real machines.

Efficiency Convention
--------------------
Round-trip efficiency is attributed entirely to the consumption side:
    η_gen = 1.0,  η_cons = η_rt
Example: GOLD η_rt = 0.782 → η_gen = 1.0, η_cons = 0.782.
This is the team convention. Do not decompose into separate gen/cons factors.

Scope
-----
- Pure Python + numpy, no Spark. Variants take arrays, not tables, so they
  run anywhere and are cheap to test.
- A variant reports revenue and the dispatch behind it, and stops there.
  What follows from that number — calibration factors, grid effectiveness,
  how an opportunity cost is finally settled — is a methodology choice and
  is left to the caller.
- The horizon is however long the price array is: 96 (24h), 192 (48h),
  672 (7d), anything. Cost grows linearly with it (LP variants) or however
  the variant scales.

Energy Only
-----------
Variants price energy and nothing else. No ancillary capacity, no
activation, no intraday alpha. For an opportunity cost this is not a gap:
those streams are the same in the free run and the curtailed run, so they
cancel in the difference between the two. Taking the delta of two
``solve()`` calls is therefore correct without them. Absolute revenue is a
different question and needs them added by the caller.

The one thing that does not cancel is ACM availability, which differs between
the two runs.

``hydra_revenue_comparison.md`` next to this file works that out against
HYDRA's fleet figures, and lists what each stream is worth.
"""

from __future__ import annotations

import numpy as np
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Fleet Configuration
# ═══════════════════════════════════════════════════════════════════════════
#
# Technical parameters for the Vattenfall German PSW portfolio.
# Source: .assistant/skills/references/fleet_reference.md
#
# Each entry is a dict that can be passed directly to a variant's solve() as
# `cfg`. The keys match every variant's expected parameter names.
#
# Key convention:
#   P_gen_max_mw   = installed generation capacity (nameplate)
#   P_cons_max_mw  = installed consumption capacity, pump mode (nameplate)
#   E_reservoir_mwh = usable reservoir capacity
#   SoC_min_mwh    = operational minimum SoC (plant-specific constraint)
#   SoC_max_mwh    = operational maximum SoC (usually = E_reservoir_mwh)
#   eta_gen        = generation efficiency (always 1.0 by convention)
#   eta_cons       = consumption-side efficiency = η_rt (round-trip efficiency)
#
# Reserve fields default to 0. Override per-day from schedule data.

FLEET: dict[int, dict[str, Any]] = {
    133: {
        "name": "GOLD",
        "description": "Goldisthal — largest German PSW, 4 turbines, 4 pumps",
        "P_gen_max_mw": 1060,
        "P_cons_max_mw": 1110,
        "E_reservoir_mwh": 9637,
        "eta_gen": 1.0,
        "eta_cons": 0.782,
        "SoC_min_mwh": 400,
        "SoC_max_mwh": 9637,
        "P_reserved_gen_mw": 0,
        "P_reserved_cons_mw": 0,
        "n_turbines": 4,
        "n_pumps": 4,
        "turbine_btags": [1, 2, 3, 4],
        "pump_btags": [43, 44, 45, 46],
    },
    134: {
        "name": "MARK",
        "description": "Markersbach — 6 turbines, 6 pumps. Max observed availability ~800 MW",
        "P_gen_max_mw": 1050,
        "P_cons_max_mw": 1140,
        "E_reservoir_mwh": 4578,
        "eta_gen": 1.0,
        "eta_cons": 0.731,
        "SoC_min_mwh": 90,
        "SoC_max_mwh": 4578,
        "P_reserved_gen_mw": 0,
        "P_reserved_cons_mw": 0,
        "n_turbines": 6,
        "n_pumps": 6,
        "turbine_btags": [6, 7, 8, 9, 10, 11],
        "pump_btags": [48, 49, 50, 51, 52, 53],
    },
    135: {
        "name": "HOH2",
        "description": "Hohenwarte 2 — never reaches nameplate (280/252 MW observed)",
        "P_gen_max_mw": 320,
        "P_cons_max_mw": 336,
        "E_reservoir_mwh": 2308,
        "eta_gen": 1.0,
        "eta_cons": 0.681,
        "SoC_min_mwh": 100,
        "SoC_max_mwh": 2308,
        "P_reserved_gen_mw": 0,
        "P_reserved_cons_mw": 0,
        "n_turbines": 2,
        "n_pumps": 2,
        "turbine_btags": [15, 16],
        "pump_btags": [57, 58],
    },
    136: {
        "name": "WEND",
        "description": "Wendefurth — smallest fleet member, 2 turbines, 2 pumps. "
                       "Discrete-unit bias expected (~25%) due to 40 MW turbine blocks",
        "P_gen_max_mw": 80,
        "P_cons_max_mw": 82,
        "E_reservoir_mwh": 531,
        "eta_gen": 1.0,
        "eta_cons": 0.752,
        "SoC_min_mwh": 5,
        "SoC_max_mwh": 531,
        "P_reserved_gen_mw": 0,
        "P_reserved_cons_mw": 0,
        "n_turbines": 2,
        "n_pumps": 2,
        "turbine_btags": [21, 22],
        "pump_btags": [63, 64],
    },
}

# Convenience lookups
AID_NAME: dict[int, str] = {k: v["name"] for k, v in FLEET.items()}
NAME_AID: dict[str, int] = {v: k for k, v in AID_NAME.items()}

# Standard Plotly colours per plant
PSW_COLORS: dict[str, str] = {
    "GOLD": "#FFDA00",
    "MARK": "#2071B5",
    "HOH2": "#4E4B48",
    "WEND": "#005C63",
}

# Evaluation months used in original backtest (H1 2025)
MONTHS: list[tuple[int, str]] = [(1, "Jan"), (3, "Mar"), (6, "Jun")]


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def _pad_or_trim(arr: np.ndarray, length: int, mode: str = "edge") -> np.ndarray:
    """Pad (by repeating last value) or trim an array to exact `length`.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    length : int
        Desired output length.
    mode : str
        numpy pad mode. Default 'edge' repeats the last value.

    Returns
    -------
    np.ndarray of shape (length,)
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)), mode=mode)


def make_curtailment_array(
    avail: np.ndarray,
    fraction: float,
    T: int | None = None,
) -> np.ndarray:
    """Build a per-QH curtailment array as a fraction of available capacity.

    This is the correct formulation for redispatch curtailment: cut from
    what's actually available, not from nameplate. A 10% curtailment on a
    plant with 800 MW available removes 80 MW, not 106 MW (nameplate).

    Parameters
    ----------
    avail : np.ndarray
        Per-QH available generation capacity (MW).
    fraction : float
        Fraction to curtail, e.g. 0.10 for 10%.
    T : int or None
        Output length. If None, uses len(avail).

    Returns
    -------
    np.ndarray
        Curtailment MW per QH (always ≥ 0).

    Example
    -------
    >>> avail = np.array([800, 800, 600, 600])
    >>> make_curtailment_array(avail, 0.10)
    array([80., 80., 60., 60.])
    """
    avail = np.asarray(avail, dtype=float)
    curtail = np.maximum(0, avail * fraction)
    if T is not None:
        curtail = _pad_or_trim(curtail, T)
    return curtail


def make_redispatch_constraints(
    start_qh: int,
    end_qh: int,
    gen_max: float = 0.0,
    cons_max: float | None = None,
) -> dict[int, dict[str, float]]:
    """Cap output over a contiguous block of quarter-hours.

    Expresses a curtailment window, e.g. 18:00-20:00, as absolute MW caps.
    Use this when the scenario is stated as an instruction. To cut a fraction
    of what the plant actually had available instead, pass ``curtail_gen`` /
    ``curtail_cons`` to a variant's solve().

    Parameters
    ----------
    start_qh : int
        First QH index of curtailment (inclusive). 0 = 00:00.
    end_qh : int
        Last QH index of curtailment (exclusive).
    gen_max : float
        Maximum generation allowed during curtailment (MW). Default 0 = full cut.
    cons_max : float or None
        Maximum consumption allowed during curtailment (MW). If None,
        consumption is unconstrained.

    Returns
    -------
    dict
        {t: {"gen_max": gen_max, ...}} for t in [start_qh, end_qh).

    Example
    -------
    >>> rc = make_redispatch_constraints(72, 80, gen_max=0)
    >>> rc[72]
    {'gen_max': 0.0}
    """
    constraints = {}
    for t in range(start_qh, end_qh):
        c = {"gen_max": float(gen_max)}
        if cons_max is not None:
            c["cons_max"] = float(cons_max)
        constraints[t] = c
    return constraints


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Price Generator (for unit tests / initial backtest)
# ═══════════════════════════════════════════════════════════════════════════

def generate_price_curves(
    seed_da: int = 42,
    seed_real: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 48h DA + realized price curves.

    Produces a realistic German winter day profile at 15-min resolution
    (hourly blocks). Used for reproducible unit tests and the original
    design-study worked example.

    Parameters
    ----------
    seed_da : int
        Random seed for DA price noise.
    seed_real : int
        Random seed for realized price deviation.

    Returns
    -------
    da_prices : np.ndarray, shape (192,)
        Day-ahead prices (€/MWh), 48h at QH resolution.
    realized_prices : np.ndarray, shape (192,)
        Realized prices = DA + systematic deviation + noise.

    Notes
    -----
    The base profile has a morning ramp (06-09), midday solar dip (10-14),
    and evening peak (17-19). Day 2 is 5% higher than day 1.
    """
    np.random.seed(seed_da)
    hourly_base = np.array([
        32, 30, 28, 27, 28, 35, 45, 58, 68, 62, 50, 42,  # 00-11
        38, 35, 37, 42, 55, 72, 85, 78, 65, 52, 42, 36,  # 12-23
    ])
    day1 = hourly_base + np.random.normal(0, 3, 24)
    day2 = hourly_base * 1.05 + np.random.normal(0, 3, 24)
    da_hourly = np.concatenate([day1, day2])
    da_prices = np.repeat(da_hourly, 4)

    np.random.seed(seed_real)
    deviation = np.random.normal(0, 4, 192)
    systematic = 3 * np.sin(2 * np.pi * np.arange(192) / 4 / 24)
    realized_prices = da_prices + deviation + systematic

    return da_prices, realized_prices
