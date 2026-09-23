"""Linear-program variant of the redispatch ``solve()`` contract.

Schedules the plant optimally against a price curve using ``scipy.optimize
.linprog``. This is the reference variant: energy arbitrage under power,
reservoir and mutual-exclusion constraints, solved to global optimality
every call. Other variants may trade that optimality guarantee for speed or
a different modelling approach, while returning the same result shape.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog
from typing import Any

from core import _pad_or_trim


# TODO: split into _build_power_bounds / _build_energy_balance /
# _build_mutual_exclusion — this body assembles four independent matrices.
def solve_lp(
    cfg: dict[str, Any],
    prices: ArrayLike,
    soc0: float,
    *,
    avail_gen: ArrayLike | None = None,
    avail_cons: ArrayLike | None = None,
    curtail_gen: ArrayLike | None = None,
    curtail_cons: ArrayLike | None = None,
    redispatch_constraints: dict[int, dict[str, float]] | None = None,
    enforce_terminal_soc: bool = False,
    terminal_soc_tol_frac: float = 0.01,
    reserve_schedule_gen: ArrayLike | None = None,
    reserve_schedule_cons: ArrayLike | None = None,
    min_stable_gen_mw: float | None = None,
    min_stable_cons_mw: float | None = None,
    eval_qh: int = 96,
) -> dict[str, Any]:
    """Find the most profitable dispatch schedule for a storage asset.

    Given a price curve and a starting reservoir level, this decides how
    much the plant should generate and how much it should consume in every
    quarter-hour, so that total revenue over the horizon is as high as it
    can be. Everything beyond ``cfg``, ``prices`` and ``soc0`` is an
    optional restriction; with none of them set the plant is limited only
    by its nameplate power and its reservoir.

    The linear program
    ------------------
    The objective is to buy in cheap quarter-hours and sell in expensive
    ones, limited by how much energy the reservoir can hold in between:

        maximise  Σ_t  price_t × (gen_t − cons_t) × Δt

    (This is often called energy arbitrage, though it is not arbitrage in
    the strict sense — nothing is bought and sold simultaneously and the
    profit is not risk-free. It is capture of the price spread between
    quarter-hours, paid for by round-trip losses.)

    The solver then builds the following restrictions on that objective.
    Each one is assembled from the arguments listed under Parameters; the
    number in brackets says which argument turns it on.

        1. Reservoir balance, every QH. Energy leaving and entering the
           reservoir must add up:

               SoC_t = SoC_{t-1} − gen_t × Δt / η_gen + cons_t × Δt × η_cons

           starting from SoC_0 = ``soc0``.

        2. Power bounds, every QH. How many MW the plant may move in each
           direction. Note this is power (MW), not stored energy:

               0 ≤ gen_t  ≤ avail_gen_t  − curtail_gen_t  − reserved_gen_t
               0 ≤ cons_t ≤ avail_cons_t − curtail_cons_t − reserved_cons_t

           [``avail_gen``/``avail_cons``, ``curtail_gen``/``curtail_cons``,
           ``reserve_schedule_gen``/``reserve_schedule_cons``]

        3. Reservoir bounds. SoC stays between ``SoC_min_mwh`` and
           ``SoC_max_mwh``, narrowed further by the energy that has to be
           kept in reserve to deliver ancillary services if called.

        4. One direction at a time:

               gen_t / P_gen_max_t + cons_t / P_cons_max_t ≤ 1

           This stops the plant generating and consuming in the same QH.
           Written as a continuous fraction rather than a binary, which
           keeps the problem an LP; the relaxation is tight for a
           single-reservoir asset, so the LP never splits the two.

        5. Terminal SoC back near ``soc0``, so the plant cannot pay for a
           good score by emptying the reservoir. [``enforce_terminal_soc``]

        6. Hard per-QH floors and ceilings from a grid operator's
           instruction. [``redispatch_constraints``]

        7. A minimum output whenever reserves are held, because a machine
           has to be running to ramp. [``min_stable_gen_mw``,
           ``min_stable_cons_mw``]

    How the bounds are stacked
    --------------------------
    Restrictions 2, 6 and 7 all land on the same per-QH bound, and they are
    applied in this order for generation (consumption is symmetric):

        1. start at ``avail_gen[t]``, or nameplate if no availability given
        2. subtract curtailment:  gen_cap = max(0, avail_gen[t] − curtail_gen[t])
        3. subtract reserves:     ub = max(0, gen_cap − reserved_gen[t])
        4. apply redispatch cap:  ub = min(ub, redispatch gen_max[t])
        5. apply min stable gen:  lb = min_stable_gen_mw, if reserves > 0
        6. feasibility guard:     lb = min(lb, ub)

    On ``curtail_gen`` versus ``redispatch_constraints``: to the optimizer a
    lost MW is a lost MW, and for a pure downward cut the two are indeed
    interchangeable. They are kept apart for two reasons. ``curtail_gen`` is
    a reduction measured against what was available, which is how a
    curtailment scenario is expressed, and it is what you want when sweeping
    "cut 10% of whatever the plant had". ``redispatch_constraints`` is an
    absolute MW instruction and is the only one of the two that can set a
    *floor* (``gen_min``/``cons_min``), i.e. a forced-operation instruction,
    which no availability cut can express.

    Parameters
    ----------
    cfg : dict
        Asset configuration. Required keys:

        - ``P_gen_max_mw``    : float — nameplate generation capacity (MW)
        - ``P_cons_max_mw``   : float — nameplate consumption capacity (MW)
        - ``E_reservoir_mwh`` : float — reservoir energy capacity (MWh)
        - ``eta_gen``         : float — generation efficiency (1.0 by convention)
        - ``eta_cons``        : float — consumption-side efficiency (= η_rt)
        - ``SoC_min_mwh``     : float — operational minimum SoC (MWh)
        - ``SoC_max_mwh``     : float — operational maximum SoC (MWh)

        Optional keys (default to 0 if absent):

        - ``P_reserved_gen_mw``  : float — scalar upward reserve (MW)
        - ``P_reserved_cons_mw`` : float — scalar downward reserve (MW)

        Use the FLEET dict entries directly, e.g. ``solve_lp(FLEET[133], ...)``.

    prices : array-like, shape (T,)
        Price curve in €/MWh. Length T determines the optimization horizon:
        T=96 → 24h, T=192 → 48h, T=672 → 7d. Any length works.

    soc0 : float
        Initial state-of-charge (MWh) at the start of the horizon.
        Use the observed reservoir level from SCADA or the start-of-day
        reading from ``asset_technical_actual_reservoirlevel_min``.

    avail_gen : array-like or None, shape (T,) or shorter
        Per-QH available generation capacity (MW) from the DA availability
        table. If shorter than T, the last value is repeated (edge-padded).
        If None, uses ``cfg["P_gen_max_mw"]`` for every QH.

        Source: ``prd_hysbap.qualified.asset_power_plan_availabilities_dayahead_qh``
        with ``type='PPmaxBr'``, aggregated from machine-level BTAGs.

        **Warning:** This table reflects DA auction availability only, NOT
        actual operational constraints. For post-DA maintenance changes,
        use ``acm_decoded_qh`` data via ``curtail_gen``.

    avail_cons : array-like or None, shape (T,) or shorter
        Same as avail_gen, for consumption capacity.

    curtail_gen : array-like or None, shape (T,) or shorter
        Per-QH generation curtailment in MW. Subtracted from available
        capacity BEFORE reserve deduction. Use ``make_curtailment_array()``
        to build this from a fractional curtailment scenario.

        Cascade: ub_gen[t] = max(0, avail_gen[t] - curtail_gen[t] - reserved[t])

    curtail_cons : array-like or None, shape (T,) or shorter
        Same as curtail_gen, for consumption capacity.

    redispatch_constraints : dict or None
        Per-QH hard bounds from a redispatch instruction. Format:
        ``{t: {"gen_max": X, "gen_min": Y, "cons_max": Z, "cons_min": W}}``
        Applied AFTER availability + curtailment + reserves. Only the keys
        present in each sub-dict are applied; others default to no override.
        Use ``make_redispatch_constraints()`` for contiguous windows.

    enforce_terminal_soc : bool
        If True, forces SoC at the last timestep to be within
        ``terminal_soc_tol_frac`` of ``soc0``. Prevents the LP from
        draining the reservoir to SoC_min (its natural tendency with
        no water-value term). Important for fair free-vs-constrained
        comparisons — without it, the free LP has an unfair advantage
        from consuming stored energy that the desk would preserve.

    terminal_soc_tol_frac : float
        Tolerance for terminal SoC as fraction of reservoir capacity.
        Default 0.01 = ±1% of (SoC_max − SoC_min).

    reserve_schedule_gen : array-like or None, shape (T,) or shorter
        Per-QH upward reserve obligation (MW). Overrides the scalar
        ``cfg["P_reserved_gen_mw"]`` when provided. Deducted from
        available generation headroom.

        Source: ``DA_PaFRRPos + DA_PFCRPos + DA_PmFRRPos`` from
        ``asset_power_plan_powerschedule_final_qh_pivoted``.

    reserve_schedule_cons : array-like or None, shape (T,) or shorter
        Per-QH downward reserve obligation (MW).

    min_stable_gen_mw : float or None
        Minimum generation (MW) when upward reserves are active.
        Models turbine minimum stable load for aFRR readiness — the
        turbine must spin at ≥ this level to ramp up if called.
        Only applied in QH where reserved_gen > 0.
        Overridden by curtailment if curtailment makes this infeasible.

    min_stable_cons_mw : float or None
        Same for the consumption side / downward reserves.

    eval_qh : int
        Number of QH over which to compute ``revenue_eval``. Default 96
        (= day 1 = 24h). The LP optimizes over the full horizon T, but
        revenue is evaluated over the first ``eval_qh`` steps only.
        This avoids end-of-horizon boundary effects contaminating the
        revenue estimate.

    Returns
    -------
    dict with keys:

        - ``status`` : str
            ``"optimal"`` if solved, ``"infeasible"`` if no feasible solution,
            or ``"infeasible_fortsetzungsfehler"`` if infeasible due to SoC
            headroom for reserves (includes a relaxed-headroom solution).
        - ``gen`` : np.ndarray, shape (T,)
            Optimal generation (MW) per QH.
        - ``cons`` : np.ndarray, shape (T,)
            Optimal consumption (MW) per QH.
        - ``soc`` : np.ndarray, shape (T,)
            State-of-charge trajectory (MWh) per QH.
        - ``net`` : np.ndarray, shape (T,)
            Net power = gen − cons (MW). Positive = feeding the grid.
        - ``revenue_total`` : float
            Total revenue over the full horizon (€).
        - ``revenue_eval`` : float
            Revenue over the first ``eval_qh`` steps (€). Use this for
            day-1 evaluation to avoid boundary effects.
        - ``revenue_per_step`` : np.ndarray, shape (T,)
            Revenue per QH (€).

        On infeasibility, only ``status`` and ``message`` are present.
        On ``infeasible_fortsetzungsfehler``, also includes
        ``relaxed_revenue_total`` and ``relaxed_revenue_eval``.

    Notes
    -----
    **SoC headroom (Fortsetzungsfehler prevention).** When the plant holds
    upward reserves (aFRR+), it must be able to ramp up generation if called.
    This requires SoC ≥ SoC_min + (P_res_gen / η_gen) × Δt at all times.
    Similarly, downward reserves require SoC ≤ SoC_max − (P_res_cons × η_cons) × Δt.
    These narrow the effective SoC band and can make the LP infeasible when
    combined with curtailment.

    **Discrete-unit bias.** This LP dispatches fractional MW. Real plants
    operate discrete turbine/pump blocks (e.g. WEND: 2×40 MW turbines).
    Small plants with few units will show higher LP-vs-reality gaps.
    WEND's expected discrete bias is ~25%.

    Examples
    --------
    Minimal call (48h, no constraints):

    >>> from core import FLEET
    >>> from variants.lp import solve_lp
    >>> prices = np.random.uniform(20, 80, 192)  # 48h
    >>> result = solve_lp(FLEET[133], prices, soc0=5000)
    >>> result["status"]
    'optimal'
    >>> result["revenue_eval"]  # day-1 revenue
    ...

    With availability, reserves, and curtailment:

    >>> cfg = FLEET[133].copy()
    >>> result = solve_lp(
    ...     cfg, prices, soc0=5000,
    ...     avail_gen=avail_gen_array,
    ...     reserve_schedule_gen=reserve_array,
    ...     curtail_gen=make_curtailment_array(avail_gen_array, 0.10),
    ...     eval_qh=96,
    ... )

    Paired-delta opportunity cost:

    >>> res_free = solve_lp(cfg, prices, soc0, avail_gen=ag)
    >>> res_curt = solve_lp(cfg, prices, soc0, avail_gen=ag,
    ...                     curtail_gen=make_curtailment_array(ag, 0.10))
    >>> opportunity_cost = res_free["revenue_eval"] - res_curt["revenue_eval"]
    """
    # ── Parse inputs ──────────────────────────────────────────────────
    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    dt = 0.25  # quarter-hour in hours

    # Scalar reserve fallbacks from config
    P_res_gen_scalar = float(cfg.get("P_reserved_gen_mw", 0))
    P_res_cons_scalar = float(cfg.get("P_reserved_cons_mw", 0))

    # Per-QH reserve arrays (override scalar if provided)
    if reserve_schedule_gen is not None:
        res_gen_t = _pad_or_trim(np.asarray(reserve_schedule_gen, dtype=float), T)
    else:
        res_gen_t = np.full(T, P_res_gen_scalar)

    if reserve_schedule_cons is not None:
        res_cons_t = _pad_or_trim(np.asarray(reserve_schedule_cons, dtype=float), T)
    else:
        res_cons_t = np.full(T, P_res_cons_scalar)

    # ── Build per-QH power bounds (the cascade) ─────────────────────────
    #
    # Step 1: Available capacity (or nameplate)
    if avail_gen is not None:
        gen_cap_t = _pad_or_trim(np.asarray(avail_gen, dtype=float), T)
    else:
        gen_cap_t = np.full(T, float(cfg["P_gen_max_mw"]))

    if avail_cons is not None:
        cons_cap_t = _pad_or_trim(np.asarray(avail_cons, dtype=float), T)
    else:
        cons_cap_t = np.full(T, float(cfg["P_cons_max_mw"]))

    # Step 2: Apply curtailment (from available, not nameplate).
    # From here on gen_cap_t / cons_cap_t are post-curtailment, pre-reserve.
    if curtail_gen is not None:
        curt_gen_t = _pad_or_trim(np.asarray(curtail_gen, dtype=float), T)
        gen_cap_t = np.maximum(0, gen_cap_t - curt_gen_t)

    if curtail_cons is not None:
        curt_cons_t = _pad_or_trim(np.asarray(curtail_cons, dtype=float), T)
        cons_cap_t = np.maximum(0, cons_cap_t - curt_cons_t)

    # Step 3: Deduct reserves → dispatchable headroom
    ub_gen_t = np.maximum(0, gen_cap_t - res_gen_t)     # gen upper bound per QH
    ub_cons_t = np.maximum(0, cons_cap_t - res_cons_t)  # cons upper bound per QH

    # ── SoC headroom for reserve delivery ────────────────────────────
    # Worst-case energy needed if reserves are called for one QH.
    # Max across the horizon (conservative: uses peak reservation).
    eta_gen = float(cfg["eta_gen"])
    eta_cons = float(cfg["eta_cons"])
    max_res_gen = float(res_gen_t.max())
    max_res_cons = float(res_cons_t.max())
    soc_headroom_gen = (max_res_gen / eta_gen) * dt if max_res_gen > 0 else 0.0
    soc_headroom_cons = (max_res_cons * eta_cons) * dt if max_res_cons > 0 else 0.0

    SoC_min = float(cfg["SoC_min_mwh"])
    SoC_max = float(cfg["SoC_max_mwh"])
    SoC_min_eff = SoC_min + soc_headroom_gen
    SoC_max_eff = SoC_max - soc_headroom_cons
    has_headroom = soc_headroom_gen > 0 or soc_headroom_cons > 0

    # ── Decision variables: [gen_0..gen_{T-1}, cons_0..cons_{T-1}, soc_0..soc_{T-1}]
    n = 3 * T

    # ── Objective: maximise revenue = Σ price × (gen − cons) × Δt ────
    # linprog minimises, so negate.
    c = np.zeros(n)
    c[:T] = -prices * dt       # gen → revenue (negate for min)
    c[T:2*T] = prices * dt     # cons → cost

    # ── Variable bounds ──────────────────────────────────────────────
    bounds: list[tuple[float, float]] = []

    # Generation bounds [0..T-1]
    for t in range(T):
        lb, ub = 0.0, float(ub_gen_t[t])

        # Min stable gen for reserve readiness
        if min_stable_gen_mw is not None and res_gen_t[t] > 0:
            lb = max(lb, float(min_stable_gen_mw))

        # Redispatch hard constraints (override)
        if redispatch_constraints and t in redispatch_constraints:
            rc = redispatch_constraints[t]
            ub = min(ub, rc.get("gen_max", ub))
            lb = max(lb, rc.get("gen_min", lb))

        # Feasibility guard: curtailment always wins over min-gen
        lb = min(lb, ub)
        bounds.append((lb, ub))

    # Consumption bounds [T..2T-1]
    for t in range(T):
        lb, ub = 0.0, float(ub_cons_t[t])

        # Min stable consumption for downward reserve readiness
        if min_stable_cons_mw is not None and res_cons_t[t] > 0:
            lb = max(lb, float(min_stable_cons_mw))

        # Redispatch hard constraints (override)
        if redispatch_constraints and t in redispatch_constraints:
            rc = redispatch_constraints[t]
            ub = min(ub, rc.get("cons_max", ub))
            lb = max(lb, rc.get("cons_min", lb))

        lb = min(lb, ub)
        bounds.append((lb, ub))

    # SoC bounds [2T..3T-1]
    for _ in range(T):
        bounds.append((SoC_min_eff, SoC_max_eff))

    # Terminal SoC constraint (optional)
    if enforce_terminal_soc:
        tol = terminal_soc_tol_frac * (SoC_max - SoC_min)
        bounds[3 * T - 1] = (
            max(SoC_min_eff, soc0 - tol),
            min(SoC_max_eff, soc0 + tol),
        )

    # ── Equality constraints: energy balance ─────────────────────────
    # SoC_t = SoC_{t-1} − gen_t × Δt / η_gen + cons_t × Δt × η_cons
    # Rearranged: (gen_t × Δt / η_gen) − (cons_t × Δt × η_cons) + SoC_t − SoC_{t-1} = 0
    # For t=0: SoC_{t-1} = soc0 (parameter, moves to RHS)
    A_eq = np.zeros((T, n))
    b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, t] = dt / eta_gen           # gen drains SoC
        A_eq[t, T + t] = -dt * eta_cons     # cons fills SoC
        A_eq[t, 2 * T + t] = 1.0            # SoC_t
        if t == 0:
            b_eq[t] = soc0                   # SoC_{-1} = initial
        else:
            A_eq[t, 2 * T + t - 1] = -1.0   # −SoC_{t-1}

    # ── Inequality constraint: mutual exclusion ──────────────────────
    # gen_t / P_gen_max_t + cons_t / P_cons_max_t ≤ 1
    # Denominator includes reserves (full machine capacity, not just headroom)
    A_ub = np.zeros((T, n))
    b_ub = np.ones(T)
    for t in range(T):
        # Denominator is capacity before reserve deduction, so the fraction stays
        # meaningful when reserves shrink the dispatchable headroom.
        pg_denom = max(float(gen_cap_t[t]), 1.0)   # avoid /0
        pc_denom = max(float(cons_cap_t[t]), 1.0)
        A_ub[t, t] = 1.0 / pg_denom
        A_ub[t, T + t] = 1.0 / pc_denom

    # ── Solve ─────────────────────────────────────────────────────────
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )

    if not result.success:
        # Diagnose: is infeasibility from SoC headroom?
        if has_headroom:
            bounds_relaxed = list(bounds)
            for t in range(T):
                bounds_relaxed[2 * T + t] = (SoC_min, SoC_max)
            retry = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds_relaxed, method="highs",
            )
            if retry.success:
                g_r, p_r = retry.x[:T], retry.x[T:2*T]
                rev_r = prices * (g_r - p_r) * dt
                return {
                    "status": "infeasible_fortsetzungsfehler",
                    "message": (
                        f"SoC headroom conflict: effective band "
                        f"[{SoC_min_eff:.0f}, {SoC_max_eff:.0f}] MWh "
                        f"too narrow for redispatch + reserves. "
                        f"Headroom: {soc_headroom_gen:.1f} MWh (gen), "
                        f"{soc_headroom_cons:.1f} MWh (cons) per QH."
                    ),
                    "relaxed_revenue_total": float(rev_r.sum()),
                    "relaxed_revenue_eval": float(rev_r[:eval_qh].sum()),
                }
        return {"status": "infeasible", "message": result.message}

    # ── Extract solution ──────────────────────────────────────────────
    gen = result.x[:T]
    cons = result.x[T:2*T]
    soc = result.x[2*T:]
    net = gen - cons
    rev_per_step = prices * net * dt

    # TODO: return a dataclass; callers index 9 untyped string keys.
    return {
        "status": "optimal",
        "gen": gen,
        "cons": cons,
        "soc": soc,
        "net": net,
        "revenue_total": float(rev_per_step.sum()),
        "revenue_eval": float(rev_per_step[:eval_qh].sum()),
        "revenue_per_step": rev_per_step,
    }
