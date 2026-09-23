# Redispatch Mechanism — Folder Structure

This folder hosts multiple *variants* of the redispatch opportunity-cost
solver. They will eventually be compared against each other, so the
structure is built around two rules:

1. **Every variant looks the same from the outside**, so callers and tests
   don't need to know which one they're calling.
2. **Nothing variant-specific leaks into the shared layer.** If it's needed
   by more than one variant, it belongs in `core.py`, not copied into each.

## Layout

```
core.py              variant-agnostic: FLEET config, array/window helpers,
                      the synthetic price generator. No solving logic.
variants/
  __init__.py         empty — `variants` is just a package.
  lp.py               the linear-program variant (the original solver).
  <name>.py           one file per additional variant, same contract.
conftest.py           shared pytest fixtures (da_prices, gold).
tests/
  helpers.py          check_invariants(), used by several test modules.
  test_core.py        tests for core.py — no solver involved.
  test_variants.py    invariants every variant must satisfy, parametrized
                      over VARIANTS. Adding a variant runs this against it
                      automatically.
  test_lp.py          things specific to the LP variant (its internal
                      mechanics, its known defects marked xfail).
run_tests.py          entry point for running the suite from Databricks.
tests_max/            notebook-based validation against real revenue data.
docs/                 write-ups of the underlying method and its concepts.
```

## The variant contract

Every variant is a function with the same signature and return shape:

```python
def solve(cfg: dict, prices: ArrayLike, soc0: float, **kwargs) -> dict:
    ...  # returns {"status", "gen", "cons", "soc", "net",
         #          "revenue_total", "revenue_eval", "revenue_per_step"}
```

`variants/lp.py::solve_lp` is the reference implementation and its docstring
documents the full result shape and the optional keyword arguments
(availability, curtailment, reserves, redispatch instructions, terminal SoC).
A new variant should accept the same `cfg`/`prices`/`soc0` and return the
same keys, even if it ignores some of the optional kwargs — that's what
lets `tests/test_variants.py` treat every variant identically.

### Adding a variant

1. Write `variants/<name>.py` with a `solve(...)` function matching the
   contract above.
2. Add it to `VARIANTS` at the top of `tests/test_variants.py`.
3. Run the test suite. The whole shared invariant suite in
   `tests/test_variants.py` now runs against it — no new tests to write.
4. If the variant has its own internal mechanics worth pinning down (not
   relevant to other variants), add a `tests/test_<name>.py` for those,
   following `tests/test_lp.py` as an example.

## Running the tests

```
python -m pytest solutions/innovations/2026_new_redispatch_mechanism -q
```

or, in Databricks, open and run `run_tests.py`. `xfailed` results are known,
tracked defects (see `tests/test_lp.py`) — not a problem. `failed` or
`xpassed` need attention.
