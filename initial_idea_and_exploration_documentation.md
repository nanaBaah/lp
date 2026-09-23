# Redispatch Flex Assets — Initial Design Exploration

This document captures the results of the unstructured initial proof-of-concept exploration done by Max in the first two
to three days of the redispatch effort, before the first team discussions took place. Some of the design decisions
described here have since evolved or been revised in subsequent meetings. Where the document presents a specific design
choice, it reflects one of several options considered at the time, not necessarily the agreed position. The purpose of
this document is to give readers a brief overview of the thought process, the experimentation results, and the open
questions that emerged from this early exploration.

The accompanying code and all numerical evidence live in
[`../tests_max/initial_poc_and_experimentation.ipynb`](../tests_max/initial_poc_and_experimentation.ipynb). This
document is the recommended entry point; the notebook is a reference to open on demand via the test IDs (T1.1, T2.3,
…) cited in each section.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Mechanism Comparison](#2-mechanism-comparison)
3. [Recommended Mechanism: How It Works](#3-recommended-mechanism-how-it-works)
4. [Key Properties](#4-key-properties)
5. [Operational Considerations](#5-operational-considerations)
6. [Open Design Questions](#6-open-design-questions)
7. [Known Limitations](#7-known-limitations)
8. [Risk Register & Defense Playbook](#8-risk-register--defense-playbook)

---

> **Notebook evidence:** Each section references tests from `initial_poc_and_experimentation` by ID (e.g. T1.1). The
> notebook TOC has the full cross-reference table.

---

## 1. Requirements

The following requirements were defined upfront to frame the design space. Any proposed mechanism must satisfy all
seven.

**1. End-to-End Process** — Complete chain: available capacity → price determination → location-/grid-constrained
    selection → activation logic → settlement.

**2. Unified Pricing & Settlement** — One model for forecast/offer and final settlement. Only input data changes
    (forecast vs actuals); methodology stays identical. No structural bias.

**3. Location-Dependent Selection** — Not purely price-driven. Must account for grid effectiveness (MW congestion relief
    per MW dispatched). Simplified DC power flow as PoC-grade proxy.

**4. Inter-Temporal & Energy-Constrained Dispatch** — Must model SoC coupling, round-trip efficiency losses, opportunity
    cost across hours. Non-negotiable for flex assets — without it, the model treats storage as small thermal plants.

**5. Transparency & Auditability** — Reproducible (same inputs → same outputs). Every selection decision explainable.
    Settlement distribution traceable.

**6. Standardisation & Configurability** — Asset-type-agnostic via JSON config. Deliverable as versioned package.

**7. Settlement Fairness** — Four criteria, priority-ranked: (1) revenue neutrality for asset, (2) cost neutrality for
    system, (3) proportional burden, (4) no cross-subsidisation.

---

## 2. Mechanism Comparison

*Notebook evidence: T2.1–T2.3 (GOLD backtest), T3.3–T3.6 (fleet validation + BCF), T4.1–T4.3 (mechanism comparison).*

Seven mechanisms were evaluated across four criteria (1–5 each, max 20).

| # | Mechanism | Description | Impl<br>*Can we build it today?* | CF<br>*Measures real cost?* | Gaming<br>*Resistant to manipulation?* | Storage<br>*Understands storage?* | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **LP Counterfactual** | LP computes optimal trajectory with/without curtailment | 3 | 3 | 4 | 5 | **15** |
| 1 | Nodal Pricing | Locational marginal prices from full grid optimisation | 1 | 5 | 5 | 4 | **15** |
| 3 | Flex Auction | Asset owners bid flexibility into a congestion market | 2 | 4 | 2 | 4 | 12 |
| 3 | Shadow Pricing | TSO publishes constraint prices from internal grid model | 1 | 4 | 4 | 3 | 12 |
| 5 | MVC (hybrid) | Settlement = max(LP path, market realisation) | 2 | 3 | 3 | 3 | **11** |
| 5 | ID Rebalancing | Early TSO instruction; asset retrades on intraday market | 4 | 2 | 3 | 2 | **11** |
| 7 | Subscription | Fixed annual capacity payment for RD availability | 3 | 2 | 3 | 2 | 10 |

### Tested: Three Mechanisms

Three mechanisms tested empirically against historical data (GOLD/MARK/WEND, H1 2025 — Jan/Mar/Jun), using
availability-steered LP with DA-declared machine capacity per QH:

1. **Test 1 — LP Counterfactual** (15/20): Standardised LP, two-stage settlement, full trajectory reoptimisation. η_pump
   gaming is self-defeating — a structurally unique advantage. Backtest Calibration Factor (BCF) calibrates for
   perfect-foresight bias. LP uses DA-declared availability per QH — not fixed nameplate capacity — which proved
   critical: MARK never reaches nameplate (800/1050 MW max gen); WEND had half capacity all of March 2025.
2. **Test 2 — ID Rebalancing** (11/20): The simplest no-model alternative. DA revenue on curtailed hours as settlement
   basis. Overpays by ~1.7× fleet-wide (range 1.3–2.8× across assets and months) because it ignores trajectory
   reoptimisation — the asset is not helpless during curtailment. Fixed-capacity LP understated this ratio;
   availability-corrected LP reveals the true gap.
3. **Test 3 — Market-Validated Counterfactual (MVC)** (11/20): LP floor + market path, settlement = max(A, B). The max()
   operator is dominated by the market path (94% of days with availability-corrected LP), making MVC functionally
   equivalent to ID Rebalancing with extra overhead.

**Not tested:** Nodal Pricing (Impl=1, requires EU-level reform — not actionable as a unilateral proposal). Flex Auction
    (Gaming=2, thin market — Vattenfall sets the price). Shadow Pricing (Impl=1, TSO won't publish). Subscription
    (Storage=2, misses trajectory entirely).

**Narrative arc:** Test 2 establishes the *need* (ID Rebalancing — settling curtailment at DA revenue on the affected
    hours, ignoring that the asset adapts its schedule — overpays by ~1.7× fleet-wide, up to 2.8× for individual
    assets). Test 1 establishes the *solution* (LP captures trajectory reoptimisation with honest, availability-steered
    capacity bounds). Test 3 closes the loop (MVC’s market path dominates on 94% of days, collapsing into the same naive
    pricing, confirming the LP's advantage).

---
## 3. Recommended Mechanism: How It Works

*Notebook evidence: T1.1–T1.2 (worked example + visualization), T6.4 (water value surface).*

*The mechanism explored in depth during this PoC. Scores: Impl 3 | CF 3 | Gaming 4 | Storage 5 | Total 15/20.*

---

### Positioning & Scope

**What:** A transparent, standardized, configurable end-to-end redispatch mechanism with opportunity-cost pricing and
    location-aware selection, designed for energy-constrained flex assets. The LP provides the first empirical,
    reproducible answer to: *"what does a PSW curtailment actually cost?"*

**Type:** Design study / regulatory proposal. Not an internal shadow process.

**Audience:** TSOs, regulators (BNetzA), and the internal Challenger Team for validation.

**Core problem:** There is no standardised methodology to compute the opportunity cost of curtailing a storage plant.
    Conceptual approaches exist (water value + efficiency losses + ancillary service value), but they leave the hard
    question unanswered: *how do you compute water value?* It depends on price expectations, reservoir state, and future
    dispatch trajectory. The LP answers this question — the shadow price of stored energy emerges from trajectory
    optimisation.

**Scope boundary:** The design study covers methodology, prototype, and scenario demonstration. It does NOT include:
    full European grid optimization, balancing market integration, aggregator governance, or production-grade software.

**Regulatory scope:** This mechanism addresses **preventive redispatch only** — congestion anticipated and planned
    ahead. Curative / emergency redispatch (real-time, response in minutes) remains under existing §13a EnWG procedures
    and is explicitly out of scope.

**Legal positioning:** This design study does not propose to replace the current §13 EnWG framework. It proposes a
    **standardised methodology** for computing the opportunity cost that §13 EnWG already entitles storage operators to
    receive. A formal legal mapping to existing energy law is identified as a required follow-on deliverable, not part
    of this technical design study. A regulatory lawyer must perform this mapping before any external presentation.

---

### Architecture Overview

The mechanism has three pillars, executed in a rolling 15-minute cycle:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  PILLAR 1: COUNTERFACTUAL LP                                                         │
│  Per-asset standardized LP (48h, 15-min resolution, public prices)                   │
│  → Computes opportunity cost = revenue_unconstrained − revenue_constrained            │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  PILLAR 2: SELECTION ALGORITHM                                                       │
│  effective_cost = opportunity_cost / grid_effectiveness                               │
│  Merit order: cheapest congestion relief first                                       │
│  Rolling re-evaluation every 15 min with updated SoC → implicit diversification      │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  PILLAR 3: SETTLEMENT                                                                │
│  Two-stage settlement:                                                              │
│    Preliminary (T+1): LP with DA prices → indicative                                 │
│    Final (T+weeks): LP with realized DA+ID prices → definitive payment               │
│  Same methodology, only inputs change. Money never settles on a forecast.             │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Counterfactual Computation

**Purpose:** Compute the opportunity cost of a redispatch activation for a single asset. The LP takes public prices, the
    plant's technical parameters, and the metered reservoir level as inputs. All inputs are either public or physically
    verifiable.

**The three-step process:**
1. **Unconstrained run**: Solve LP freely → `Revenue_free`
2. **Constrained run**: Solve LP with redispatch instruction as additional bound (e.g., `gen_t ≥ X` or `gen_t ≤ Y`) →
   `Revenue_constrained`
3. **Opportunity cost** = `Revenue_free − Revenue_constrained`

**Key design choices:**

| Parameter | Value | Rationale |
| --- | --- | --- |
| Time resolution | 15 min | German market granularity |
| Optimization horizon | **E/P-coupled** (settle first 24h) | Calibrated formula: `horizon_h = max(48, round_4h(E/P × 6))`. Zero-gap horizon for GOLD (E/P=9.1h): **54.4h ≈ 56h** (6× E/P). Fleet: GOLD 56h, all others 48h. **Horizon is NOT neutral** — backtest sweep (GOLD, March 2025, 7 horizons 48–72h): 48h overestimates +4.6%, 56h underestimates −2.4%, 72h underestimates −5.3%. Gap drops discontinuously between 52h (+3.8%) and 56h (−2.4%) — driven by specific cross-day price patterns, not smooth decay. Default 48h is incentive-compatible (favors asset → encourages participation). 56h is the calibrated neutral point. Horizon is a configurable parameter per asset class. |
| Foresight | Perfect (deterministic) | Justified: final settlement uses realized prices |
| Price inputs | EPEX DA (hourly, flat within 15-min steps) + published ID indices | Public. DA prices are hourly — mapped to 15-min steps as flat blocks. ID indices at best available granularity. |
| Solver | Open-source (HiGHS, scipy, PuLP) | Transparency requirement |
| Price-taker | Yes (for PoC) | Simplification; market impact modeling deferred |

---

### Selection Algorithm

**Core rule:** Rank assets by effective redispatch cost, activate in merit order.

```
effective_cost_{i,t} = opportunity_cost_{i,t} / grid_effectiveness_{i,t}
```

Interpretation: EUR per MW of congestion relief. Select the cheapest relief first.

**Grid effectiveness:** From simplified DC power flow model (PoC-grade reduced network). Measures how many MW of
    congestion relief 1 MW from asset *i* delivers at the constrained line.

**Rolling re-evaluation loop (every 15 min during congestion):**

```
For each candidate asset:
    1. Read current SoC (metered)
    2. Re-run counterfactual LP with updated SoC + current prices
    3. Compute effective_cost

Rank by effective_cost (ascending)
Activate in merit order until required relief volume is met
    → Partial activation allowed (fill incrementally)
Record: selected assets, volumes, scores, SoC before/after
```

**Inter-temporal handling:** No joint multi-asset optimization needed. The LP's 48h horizon means that depleting an
    asset raises its shadow price naturally. The rising cost pushes the merit order toward other assets —
    diversification is emergent, not engineered.

**Example:**

| Asset | Opportunity cost (€/MWh) | Grid effectiveness | Effective cost (€/MW relief) | Rank |
| --- | --- | --- | --- | --- |
| PSW A | 80 | 0.90 | 89 | 1st ✅ |
| Battery C | 100 | 0.95 | 105 | 2nd |
| PSW B | 60 | 0.50 | 120 | 3rd |

PSW B looks cheapest by raw price but delivers least relief per euro. The effective-cost ranking corrects for this.

---

### Settlement

**Two-stage settlement (preliminary → final):**

| Stage | Timing | Price input | Purpose |
| --- | --- | --- | --- |
| Preliminary | T+1 (next business day) | DA prices | Cash flow signal, indicative |
| Final | T + weeks/months | Realized DA + ID prices | Definitive compensation |

**Settlement amount per activation event per asset:**
```
Settlement = Σ_t [ (Revenue_free_t − Revenue_constrained_t) ]    for all t in activation window
```
Each 15-min slot is settled individually, but the LP that computes each slot's value is 48h forward-looking. The
settlement unit is 15 min; the economic methodology is inter-temporal.

**Same methodology, different inputs:** Preliminary and final runs use identical LP code and parameters. Only the price
    curve changes (DA forecast → realized prices). Satisfies the unified model requirement.

---

## 4. Key Properties

*Notebook evidence: T5.4 (gaming prevention), T6.2–T6.3 (generalizability + scale invariance).*

### Fairness Framework

Four properties, in priority order:

| Priority | Property | Measurable definition |
| --- | --- | --- |
| 1 (anchor) | **Revenue neutrality for asset** | Post-settlement, asset's financial position ≥ counterfactual position (no loss from participation) |
| 2 | **Cost efficiency for system** | Total redispatch cost ≤ today's total cost for same congestion relief |
| 3 | **Proportional burden** | If multiple assets are curtailed, financial impact is proportional to each asset's contribution to relief |
| 4 (constraint) | **No cross-subsidization** | One asset's activation does not improve another asset's settlement position |

---

### Gaming Prevention

Of four identified attack vectors, three are eliminated by architecture:

| Vector | Status | Mechanism |
| --- | --- | --- |
| Price forecast gaming | **Eliminated** | Only public prices as LP inputs |
| Strategic scheduling | **Eliminated** | Counterfactual is model-based, not schedule-based |
| Availability manipulation | **Eliminated** | SoC is metered/SCADA-verified at each 15-min step |
| Parameter inflation | **Mitigated** | Auditable parameter registry; physical verifiability; LP cross-validation against historical dispatch |

---

### Generalizability: Beyond Pumped Storage

The mechanism's value is proportional to the asset's **inter-temporal coupling** — the degree to which curtailing one
hour disrupts the optimal dispatch trajectory across other hours. Assets with an energy constraint (SoC) have strong
coupling; stateless assets have none.

**There is no "battery vs PSW" distinction.** The LP is identical. Controlled sweeps (P=10–1000 MW) confirm **perfect
    scale invariance**: a 10 MW community battery and a 1000 MW PSW with the same E/P and η produce the same
    Reoptimisation Multiple (Δ = 0.0000). The ratio is governed by three factors:

1. **E/P ratio** — primary driver (lower E/P → higher ratio, 4.9× at 1h, converging to ~2× at ≥8h)
2. **η_rt** — secondary (lower η slightly increases the ratio; effect negligible above 85%)
3. **Price curve & curtailment window** — context-dependent (ratio varies with which hours are curtailed)

| Asset type | E/P | Reoptimisation Multiple | Mechanism value |
| --- | --- | --- | --- |
| **Any SoC-coupled asset** | 1h | 4.9× | Essential — short-duration assets are the *most* distorted by naive methods |
| | 2h | 3.9× | Essential |
| | 4h | 2.4× | Essential — typical for medium BESS and some PSWs |
| | 8h | 2.0× | Essential — converges; matches PSW GOLD backtest (2.0× on real data) |
| | 12h | 1.9× | Essential — plateau; very large reservoirs see diminishing marginal disruption |
| **Solar + battery hybrid** | 4h (bat) | 1.9× | Essential — solar inflow partially buffers battery trajectory; slightly lower ratio |
| **Gas CCGT (min-load)** | n/a | ~1.1× | Marginal — inter-temporal coupling limited to startup/shutdown |
| **Stateless (OCGT, solar, wind)** | n/a | 1.0× | Unnecessary — OC = price × curtailed volume; no reoptimisation possible |

**Solar + battery hybrids** require one config extension: a `solar_profile_mw` array as an exogenous generation input.
    The LP treats solar output as an uncontrollable inflow (the battery can charge from it but cannot exceed it). The
    solar inflow acts as a partial buffer, slightly reducing the Reoptimisation Multiple compared to a standalone
    battery at the same E/P.

**Stateless assets** (gas OCGTs, pure renewables) can use the LP correctly, but it degenerates to simple multiplication:
    `OC = (price − marginal_cost) × curtailed_MW × Δt`. The mechanism is correct but adds no value over existing
    cost-based approaches.

**Regulatory framing:** One unified, asset-agnostic framework. The LP is parameterised by (P, E, η, SoC bounds) —
    technology label is irrelevant. The mechanism's advantage scales with E/P, which is exactly the dimension where
    simple cost-based compensation fails most.

**BCF applicability:** BCF ≈ 1.0 for stateless assets (LP = actual, no optimisation gap). For storage and hybrids, BCF
    varies by E/P ratio and trading desk skill, as demonstrated in the cross-fleet backtest.

---

## 5. Operational Considerations

*Notebook evidence: T3.2 (availability audit / feasibility).*

Key operational questions explored during the PoC.

### Feasibility & Obligation Protection

**"How much can we do?"** Before accepting a curtailment instruction, the LP verifies that compliance is physically
    feasible given the current reservoir level and existing balancing obligations (aFRR, mFRR, FCR). Two layers of
    protection are built in:

**Layer A — Capacity reservation.** The LP reduces available generation and pump capacity by the amount reserved for
    balancing obligations. If the plant has committed 100 MW of upward reserve, the LP can only offer generation
    capacity above that commitment. Same for pumping and downward reserve.

**Layer B — Reservoir headroom.** The LP raises the minimum allowed reservoir level so the plant always has enough
    stored energy to deliver upward reserve if called, and lowers the maximum so it can absorb downward reserve. If a
    curtailment instruction would push the reservoir into a zone where these obligations cannot be met, the LP returns
    `infeasible_fortsetzungsfehler` with a diagnostic flag — telling the TSO: *this asset cannot accept this curtailment
    without compromising its balancing commitments.* The TSO then selects a different asset.

**Proposed extension — Availability envelope.** Each plant should proactively publish a redispatch availability envelope
    every 15 minutes: *"given my current reservoir, prices, and obligations, I can provide at most X MW of generation
    curtailment during hours Y–Z."* The TSO uses this to pre-filter candidates before running the merit order — avoiding
    trial-and-error (try a curtailment, get “infeasible”, try another plant). The LP can compute this envelope by
    running a sweep of curtailment levels and finding the feasibility boundary.

### Independence from Intraday Retrading

When a curtailment instruction arrives, the trading desk retrades on the intraday market: buys back short positions on
curtailed hours, sells generation on adjacent hours, adjusts pumping. This retrading IS the trajectory reoptimisation
the LP computes.

**The settlement is independent of what the desk actually does.** The LP computes the theoretical optimal trajectory
    with and without curtailment. The difference is the compensation. Whether the desk retrades brilliantly or does
    nothing at all does not change the settlement for that event.

**The calibration factor absorbs the gap.** The rolling 12-month BCF measures how close the desk’s actual revenue comes
    to the LP optimum. If the desk is good, BCF is high and future settlements are larger. If the desk is poor, BCF
    drops and future settlements shrink. This is **incentive-compatible**: the plant has a financial reason to retrade
    well, but the mechanism does not require or depend on it. Because BCF is a 12-month average, a single bad or good
    day does not create swings.

**Key principle:** The desk should neither be penalised nor rewarded for its specific response to a curtailment event.
    The settlement is model-based, not behaviour-based.

### Settlement Timing

The option explored uses start-of-day conditions: metered midnight reservoir level, DA-declared machine availability,
and public prices. Using midnight SoC avoids gaming vectors (plant positioning reservoir to inflate settlement, or TSO
timing orders to minimise payments). The same curtailment on the same day always produces the same settlement,
regardless of when the order was placed.

### Bilateral Execution

Both the TSO and the plant operator run the same open-source LP with the same public inputs. If both run it with the
same data, they get the same answer. Disputes reduce to factual questions about inputs, not methodology.

---

## 6. Open Design Questions

*Notebook evidence: T5.1 (8 questions overview), T3.1 (terminal SoC — §6.5), T5.5 (cascading + pump curtailment —
§6.1/§6.3), T5.2–T5.3 (aFRR/min-gen — §6.4), T5.9 (curtailment timing), T5.10 (pump curtailment + forced pumping —
§6.3).*

Eight questions identified during exploration. Items 1–4 are hard gaps (structural decisions needed). Items 5–8 are
medium gaps (condensed below).

### 6.1 Multi-Day Consecutive Curtailment

**Severity: High. Most likely real-world scenario.**

If the TSO curtails Monday, Tuesday, Wednesday, the midnight SoC on Tuesday is distorted by Monday’s curtailment. The
plant enters Tuesday with a suboptimal reservoir position — not because of its own trading decisions, but because of
Monday’s grid operator instruction.

**Option explored:** Each day settles independently from its **actual metered midnight SoC.** This is the correct choice
    for three reasons:

1. **Simplicity.** The LP stays single-day (48h). No multi-day optimisation, no state-coupling across settlement
   periods.
2. **Correctness.** Tuesday’s counterfactual asks: “given where the reservoir actually is at midnight Tuesday, what is
   the cost of today’s curtailment?” This is the right question — the plant’s actual starting state is a fact, not a
   counterfactual.
3. **Monday’s compensation already covered Monday’s damage.** If Monday’s curtailment left the reservoir 500 MWh lower
   than optimal, Monday’s settlement included the cost of that displacement. Tuesday’s LP starts from the actual state
   and computes Tuesday’s incremental damage.

**Residual risk:** If the reservoir is driven to a corner (near SoC_min or SoC_max) by consecutive curtailments, the
    LP’s Tuesday counterfactual may understate the cumulative damage — because the “unconstrained” revenue from a
    depleted reservoir is lower than it would have been from a well-positioned reservoir. The Monday settlement captured
    the first-order effect, but not the second-order “reduced optionality on Tuesday because Monday’s curtailment left
    the reservoir in a worse starting position.”

**Mitigation options (for future refinement, not PoC):**
* **SoC reset compensation.** If the metered midnight SoC deviates from the LP’s predicted end-of-day SoC by more than a
  threshold (e.g., 5% of E_reservoir), the difference is valued at the average water value (LP shadow price) and added
  to Monday’s settlement. This compensates the position damage explicitly.
* **Multi-day LP.** Extend the optimisation horizon to cover the full curtailment episode (e.g., 72h for a 3-day event).
  Settles the entire episode as one event. More accurate but significantly more complex.

**PoC position:** Independent daily settlement. Flag the residual risk. Quantify the second-order effect in the backtest
    as a future deliverable.

### 6.2 Stacking Multiple Instructions in One Day

**Severity: High. Inevitable in practice.**

The TSO curtails 18:00–20:00 for congestion on line A, then at 14:00 adds 16:00–17:00 for line B. The combined
opportunity cost is **not** the sum of the individual opportunity costs — the second instruction’s cost depends on the
first already being in place. The LP with both constraints simultaneously may produce a higher or lower OC than the sum
of two independent LP runs.

**Option explored:** One combined LP run per delivery day. All active curtailment instructions are applied
    simultaneously as constraints. The LP computes:
* `Revenue_free` — no curtailment
* `Revenue_constrained` — all active instructions combined
* `OC_total = Revenue_free − Revenue_constrained`

**Attribution (if required by regulation):** If different congestion elements have different cost allocation rules
    (e.g., line A costs to TSO region North, line B to TSO region South), compute the marginal contribution of each
    instruction:
* `OC_A = Revenue_free − Revenue_constrained_A_only`
* `OC_B = Revenue_free − Revenue_constrained_B_only`
* `OC_combined = Revenue_free − Revenue_constrained_AB`
* Interaction term: `OC_combined − OC_A − OC_B` (can be positive or negative)
* Allocate interaction term proportionally: `share_A = OC_A / (OC_A + OC_B)`, etc.

This Shapley-value-like decomposition is exact for two instructions and computationally tractable for small numbers. For
many simultaneous instructions (unlikely for a single asset), use the combined run and split costs by curtailed-MWh
share.

**PoC position:** Combined LP run. No per-instruction attribution in the prototype. Flag as a refinement for production.

### 6.3 Upward Redispatch (Forced Generation)

**Severity: High. Half the mechanism is missing.**

§13 EnWG covers both directions: the TSO can order an asset to **stop generating** (downward redispatch / curtailment)
or to **start generating** (upward redispatch / forced dispatch). The current design only addresses curtailment.

For a PSW, forced generation means discharging the reservoir at times when the trading desk would rather pump or sit
idle — typically low-price hours. The cost structure is inverted:

* **Curtailment** blocks high-value discharge → the reservoir stays fuller than optimal → the plant shifts to less
  profitable hours. Cost = lost peak revenue minus recovered off-peak revenue.
* **Forced generation** depletes the reservoir at low-value hours → the reservoir ends up emptier than optimal → the
  plant has less energy to sell at future high prices. Cost = low revenue received now + lost future high-price revenue
  + pumping cost to refill.

**LP treatment:** Identical to curtailment, with the constraint direction reversed: `gen_t ≥ X` instead of `gen_t ≤ Y`.
    The LP computes `Revenue_free − Revenue_constrained` as before. The key difference is operational:

* Forced generation with a **nearly empty reservoir** is extremely expensive — the plant must pump first (at whatever
  price) to have energy to discharge, then loses the energy at a low price. The LP captures this through the SoC
  constraints.
* Forced generation with a **nearly full reservoir** is cheap — the plant was going to discharge soon anyway; the TSO
  just moved it to a less optimal time.
* If the reservoir is too empty to comply, the LP returns `infeasible` — the plant physically cannot generate.

**Settlement:** Same two-stage process. Same BCF calibration. The OC can be large for forced generation during off-peak
    with low SoC, and near-zero for forced generation during peak with high SoC.

**Communication note:** Upward redispatch compensation can be negative from the plant’s perspective in rare cases — if
    the forced generation happens to coincide with what the plant would have done anyway. The settlement should allow
    for zero or near-zero OC without creating disputes.

### 6.4 Ancillary Service Interaction

**Severity: Upgraded from Medium to High based on empirical testing.**

Two distinct effects:

**Effect A — Lost reservation revenue.** If curtailment forces the plant to withdraw its reserve commitment, the lost
    reservation payment is an opportunity cost not captured by the energy-only LP. **Empirical finding (GOLD March
    2025): negligible.** At €5/MW/h with 53 MW mean reservation and 2h curtailment, the lost revenue is €534 — 0.4% of
    the energy OC. Even at €20/MW/h: 1.4%. A separate line item in the settlement is sufficient; co-optimisation adds
    complexity for almost no money.

**Effect B — Minimum stable generation for reserve readiness.** This is the larger issue. When the plant holds upward
    aFRR, at least one turbine must be spinning at minimum stable load (~50 MW per unit). The LP did not enforce this —
    it could set `gen_t = 0` while claiming capacity was “reserved”, which is physically impossible.

**Empirical findings (GOLD, March 2025, 31 days):**

| Metric | Without min-gen | With min-gen |
| --- | --- | --- |
| LP monthly revenue | €15,142,438 | €14,522,235 |
| BCF (actual ÷ LP) | 0.956 | **0.996** |
| Mean OC per event | €143,574 | €151,852 (+9.4%) |
| Calibrated monthly OC | €4,253,340 | **€4,690,694 (+10.3%)** |
| QHs where min-gen binds | 43/96 (free), 41/88 (curt) | — |

**Key insight:** Min-gen does NOT cancel in the counterfactual. The curtailed run has 8 fewer QHs subject to min-gen
    (because curtailment overrides reserve readiness at `gen_max=0`), so the free run is slightly MORE constrained
    relative to the curtailed run. This increases the raw OC by +9.4%. Combined with the BCF improvement (0.956 →
    0.996), the calibrated settlement increases by +10.3% (€437k/month for GOLD alone).

**Why it matters:** The standard LP *undercompensates* the plant by overstating its theoretical flexibility. Adding
    min-gen makes the LP more realistic (BCF ≈ 1.0), which increases the settlement. This is the correct outcome — the
    plant IS less flexible than the standard LP assumes, and curtailment costs MORE for a less-flexible plant.

**LP extension (implemented):** The solver now accepts:
* `min_stable_gen_mw` — minimum generation when upward reserve is active
* `min_stable_pump_mw` — same for downward reserve
* `reserve_schedule_gen` / `reserve_schedule_pump` — per-QH reserve arrays

Curtailment constraints override min-gen (lb = min(lb, ub)), correctly modelling that the TSO’s instruction takes
priority over reserve readiness.

**PoC position: Include min-gen in the PoC LP.** The +10.3% calibrated settlement change and BCF improvement from 0.956
    to 0.996 are both material. This is not a production refinement — it belongs in the prototype. Reservation revenue
    (Effect A) remains a separate line item.

---

### 6.5 Further Questions Explored

* **Negative prices.** When DA prices go negative, curtailment during pumping hours could benefit the plant. The option
  explored was to floor opportunity cost at zero for settlement, preventing perverse incentives (TSO profiting from
  curtailing assets during negative-price hours).
* **Terminal SoC boundary.** The LP optimises over 48h but settles only 24h. Without a terminal constraint the LP can
  game the horizon boundary. The option explored was `soc_T = soc_0` (return to starting level) in both runs. Material
  for short-duration assets (E/P ≤ 2h), negligible for large reservoirs.
* **Forward positions vs spot prices.** Stakeholders will ask whether compensation is based on forward or spot prices.
  The option explored was spot-only: forward contracts settle financially regardless of physical dispatch, and using
  forward prices would require access to proprietary hedge books.
* **SoC measurement tolerance.** Converting reservoir water level to MWh has measurement uncertainty (±2% for GOLD =
  ±193 MWh ≈ ±€10k per event). Needs an agreed conversion curve and dispute threshold in any production protocol.

---

## 7. Known Limitations

*Notebook evidence: T3.4 (out-of-sample BCF), T5.6 (BCF stability defense), T5.8 (seasonal BCF), T6.1 (cost
quantification).*

| Limitation | Severity | Status |
| --- | --- | --- |
| **Legal framework (§13 EnWG) not mapped** | **High** | Open. Regulatory lawyer must map to existing law. Blocks external presentation. |
| **TSO grid model acceptance** | **High** | Open. DC model is PoC only. Production requires TSO-provided effectiveness factors. |
| **Cost pass-through (→ Netzentgelte)** | **Medium** | LP computes €74/MWh fleet-wide; annual total depends on activation frequency (Challenger Team deliverable). |
| **ID trading value gap** | **Medium** | Accepted as conservative. LP undervalues intraday alpha; future ID adjustment factor possible. |
| **European grid coupling** | **Medium** | PoC simplification. Ties to TSO grid model. |
| **LP perfect-foresight bias** | **Medium** | Addressed. BCF calibration from 12-month rolling backtest corrects cross-fleet. |
| **48h horizon for large PSWs** | **Medium** | Addressed. E/P-coupled formula: GOLD gets 56h, fleet stays 48h. |
| **DA prices hourly, not 15-min** | **Low** | Accepted. Flat within-hour blocks; ID prices in final settlement partially compensate. |
| **Price-taker assumption** | **Low** | PoC simplification. Market impact modeling deferred. |
| **Rolling vs joint optimization** | **Low** | Gap is small due to LP's 48h foresight. Full portfolio optimization is a refinement option. |

---

## 8. Risk Register & Defense Playbook

*Notebook evidence: T3.2 (phantom capacity safeguard), T5.4 (η_pump gaming), T5.6 (BCF safeguard), T5.7 (OC stability /
difference-cancels-bias).*

*Adversarial review from the perspective of regulators (BNetzA), TSOs, and competing asset owners.*

---

### Threat Model

| Persona | Motivation | Attack style |
| --- | --- | --- |
| **BNetzA regulator** | Legal mandate, consumer protection | "Show me the legal basis. Who verified the counterfactual?" |
| **TSO (50Hertz, TenneT, Amprion, TransnetBW)** | Operational sovereignty, liability, existing processes | "Our grid model says otherwise. We can't accept third-party power flow." |
| **Other asset owners (thermal)** | Equal treatment | "Why does storage get opportunity-cost while we're stuck with cost-based?" |

---

### Open Risks

| ID | Risk | Severity | Mitigation path |
| --- | --- | --- | --- |
| V2 | **Legal framework (§13 EnWG) not mapped** | **High** | Regulatory lawyer must map each component to §13 EnWG or argue for framework amendments. Non-negotiable prerequisite for any external presentation. |
| V3 | **TSOs will not accept third-party grid models** | **High** | The simplified DC model is a demonstration tool only. Production deployment requires TSO-provided grid effectiveness factors (PSS/E, INTEGRAL). The proposal's value is the framework, not the grid model. |
| V6 | **Cost pass-through (→ Netzentgelte)** | **Medium** | Unit costs quantified: LP €74/MWh (fleet avg), naive approaches claim €145/MWh, thermal fallback €150–400/MWh. Annual fleet cost depends on activation frequency (Challenger Team must supply). |
| V9 | **European grid coupling not modeled** | **Medium** | PoC simplification. Production deployment requires TSO-provided factors that inherently include cross-border effects. |

---

### Built-In Safeguards

The following risks were identified during design and are addressed in the specification:

| Risk | Safeguard (see Design Specification) |
| --- | --- |
| Simultaneous gen+pump inflates counterfactual | Linear mutual exclusion constraint: `gen_t/P_gen_max + pump_t/P_pump_max ≤ 1` |
| Balancing commitments excluded → infeasible LP | `P_reserved_gen_t` / `P_reserved_pump_t` as capacity constraints |
| Fortsetzungsfehler (SoC-commitment conflict) | LP enforces SoC headroom for reserve delivery; returns `infeasible_fortsetzungsfehler` diagnostic if violated |
| LP uses phantom capacity (maintenance bias) | LP steered by DA-declared machine availability per QH, not fixed nameplate. Without this, BCF is depressed by 11–19 pp across the fleet. |
| LP overestimates (perfect foresight bias) | Backtest Calibration Factor (BCF): `Calibrated_OC = LP_OC × BCF` from rolling 12-month backtest. Validated cross-fleet with availability-steered LP. |
| η_pump parameter gaming | Self-defeating by design (OC maximised at true η). ±2% SCADA tolerance + annual audit. |
| 48h horizon insufficient for large PSWs | E/P-coupled horizon formula: `max(48, round_4h(E/P × 6))`. GOLD gets 56h; fleet stays at 48h. |
| Scope ambiguity (curative vs preventive) | Explicitly scoped to preventive redispatch only. Curative remains under §13a EnWG. |

---

### Blocking Prerequisites

| # | Action | Owner | Blocks |
| --- | --- | --- | --- |
| 1 | **Regulatory legal mapping (§13 EnWG)** | External counsel | External presentation |
| 2 | **Challenger Team: validate activation frequency** | Challenger Team | Annual cost estimate (unit costs computed, frequency unknown) |
| 3 | **TSO engagement on effectiveness factors** | Business Development / Asset Mgmt | Production credibility |


