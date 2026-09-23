# Fair Compensation for Storage Plants During Grid Curtailment

When the electricity grid is congested, the grid operator intervenes by curtailing
power plants. This can mean ordering a plant to reduce its generation, or in some
cases forcing it to consume more. These plants must be compensated for their lost
income. For a gas plant this is straightforward: the lost income equals the profit
margin on the fuel it would have burned. For a storage plant it is not, as storage
has no fuel cost.

But storage does earn real money. It buys electricity when prices are low, pumps
water uphill to store it, then releases that water to generate electricity when
prices are high. The profit comes from the gap between those prices. When a storage
plant is told to stop generating during expensive hours, or forced to pump when it
would rather wait, it does not just lose those hours. The restriction changes the
plant's entire earnings plan for the day, because every hour of charging and
discharging is connected.

The true cost comes down to one question: how much would the plant have earned
today without the restriction? The gap between its unrestricted earnings and its
curtailed earnings is the fair compensation. This work focuses on quantifying that
changed-schedule cost.

---

## Three types of cost

When a storage plant is curtailed, three distinct costs arise:

1. **Changed schedule cost.** The plant's optimal plan for the remainder of the
   day is disrupted. Generation, pumping, and reservoir management must all be
   rearranged, reducing the total profit the plant can extract over the horizon
   still ahead of it. Valuation therefore uses the price curve that is relevant
   at the moment the restriction becomes known: day-ahead prices for a
   curtailment announced before the day-ahead auction, and the applicable
   intraday price curve for one announced during the delivery day. This is the
   largest component and the focus of this work.
2. **Lost intraday trading opportunity.** Beyond its physical schedule, a storage
   plant can trade on the intraday market for the same delivery hours without
   changing its actual dispatch. Curtailment reduces or removes this opportunity.
   This cost is real but not quantified here.
3. **Wear cost.** Curtailment may force the plant into a different operating
   pattern with additional starts, stops, or mode changes, causing mechanical
   wear. This cost is real but small relative to the other two and not quantified
   here.

---

## What a redispatch mechanism needs to do

Existing compensation rules were designed around conventional generators and do
not capture how storage plants create and lose value. Rather than patching
existing compensation rules, we develop a method from first principles. A
complete redispatch mechanism for energy-constrained flex assets must satisfy
seven requirements:

1. **End-to-end process.** Cover the complete chain from available capacity
   through price determination, grid-constrained selection, activation logic,
   to settlement.
2. **Unified pricing and settlement.** One model for both the forecast/offer
   stage and the final settlement. Only the input data changes (forecast vs
   actuals); the methodology stays identical. No structural bias between the
   two.
3. **Location-dependent selection.** Selection cannot be purely price-driven.
   It must account for grid effectiveness: how much congestion relief each MW
   of curtailment actually delivers, which depends on where the plant sits in
   the grid.
4. **Inter-temporal and energy-constrained dispatch.** The model must capture
   how reservoir level, round-trip efficiency losses, and opportunity cost
   connect hours to each other. Without this, storage is treated like a small
   thermal plant, and the compensation will be wrong.
5. **Transparency and auditability.** The method must be reproducible (same
   inputs produce the same outputs). Every selection decision must be
   explainable and every settlement figure traceable.
6. **Standardisation and configurability.** The method must work across asset
   types through configuration (e.g. a JSON parameter set), not through custom
   code per plant. Deliverable as a versioned package.
7. **Settlement fairness.** Four criteria in priority order: (1) revenue
   neutrality for the asset owner, (2) cost neutrality for the system,
   (3) proportional burden sharing, (4) no cross-subsidisation between assets.

---

## The proposed approach for changed schedule cost

### Step 1: Establish what the plant could have earned at most

To measure what the plant lost, we first need to know what it could have earned
at most without curtailment. We do this by finding the theoretically optimal
schedule: given the electricity prices and the plant's technical constraints
(reservoir size, pump and generation capacity, round-trip efficiency), what
dispatch plan would have maximised revenue?

The specific method is an implementation choice, not the core of the idea. It
could range from a simple heuristic that allocates generation to the most
expensive hours to a full mathematical optimisation model — any approach that
produces a realistic dispatch schedule will work. What matters is that the
method is applied identically in steps 1 and 2, so that modelling errors cancel
in the difference.

We propose a mathematical optimisation model (linear programme, LP) as our
recommended approach: given prices and constraints, in our implementation it
would find the dispatch schedule that maximises revenue over a rolling 48-hour
window in 15-minute steps, settling on the first 24 hours. The second day acts
as a lookahead buffer, preventing the model from artificially draining the
reservoir at the end of the horizon. It gives an exact answer, every step is
auditable, and it runs in milliseconds. The prices it optimises against are the
real, observed price curve applicable at the time of the event — no price
forecast is involved. The idealisation lies purely on the dispatch side: the
model extracts the maximum value that this price curve allows, which a real desk
would not necessarily have captured. It is therefore a normative benchmark, not
a prediction of what the desk would actually have traded.

### Step 2: Compute the cost of curtailment

Run the same simulation again, but this time with the curtailment constraint in
place: generation capacity is reduced to zero (or capped) during the restricted
hours. The plant will adapt its plan around the restriction, recovering some of
the lost value. The difference between the uncurtailed and curtailed profits is
the true opportunity cost.

This "paired comparison" has an important property: any systematic errors or
simplifications in the simulation appear on both sides and cancel out in the
subtraction. The simulation does not need to be a perfect revenue predictor —
it only needs to correctly measure the *difference* that curtailment makes.

With the LP, this step is particularly powerful: the model does not simply zero
out the restricted hours. It re-optimises the entire day's dispatch from
scratch, shifting generation and pumping across all remaining hours to recover
as much value as possible. This captures the cascading effect that makes storage
different from thermal plants — curtailment in one hour changes the optimal use
of every other hour.

---

## Core assumptions

1. **Ideal-dispatcher benchmark on a real price curve.** The price curve used is
   the actual one applicable at the time of the event; nothing is forecast. What
   is idealised is the dispatch decision: compensation is measured against the
   best schedule that curve permits, not against what a specific trading desk
   would have achieved. A real desk deviates from this benchmark for two
   reasons: it may simply execute imperfectly (the minor effect), and it may act
   on information the model does not hold — fundamental views, risk limits,
   portfolio positions, cross-market opportunities (the material effect).
2. **Symmetric application.** The same method, parameters and data vintage are
   used for the free and the curtailed run, so systematic modelling error
   cancels in the difference.

---

## Open points / decisions to be taken

1. **Optimisation horizon.** The draft assumes a rolling 48-hour window settled
   on the first 24 hours, with day 2 as lookahead buffer. 48h is a placeholder —
   the horizon length (24h / 48h / longer) is still open and drives how much
   inter-temporal value the counterfactual can capture.

