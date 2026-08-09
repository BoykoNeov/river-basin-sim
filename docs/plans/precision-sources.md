# Precision pass — compensated accumulation for areal sources

**Status: done, 2026-08-09.** One of the two debts M6 measured and deliberately did
not fix (`roadmap.md` → "Carried debts before M7"). Not a milestone: a scoped
correctness pass on one arithmetic path, landing before M7 puts a second
distributed source onto the same fields.

---

## 1. The defect

Rain is an **areal** source: every cell in the domain receives `rate*dt` metres,
every step, for as long as the storm lasts. In the solver that is one float32 add:

```
h[i,j] = h[i,j] + rate*dt
```

At M6's reach demo the increment is `rate*dt ≈ 8e-5` m onto a depth already at
`0.1 .. 1.4` m. `eps(1.0)` in float32 is `1.2e-7`, so each add discards a few tens
of nanometres. The discarded part does **not** average to zero over the population,
and a couple of hundred thousand cells × a few hundred storm steps turns it into a
measurable volume of water that the ledger — which banks rain analytically in
float64 — correctly reports as missing.

This is HANDOFF §12's drift-at-scale, and M6 measured it rather than guessing:
`scenarios/reach_basin.toml` at `coarsen = 4` **exceeded the 1e-6 mass gate**, and
did so **identically with sub-grid channels disabled**, which is what established it
as arithmetic rather than new code.

### The evidence that it is the source, not the fluxes

From the failing run's ledger series (residual, m³):

| t (s) | residual | note |
|---|---|---|
| 1800 | −3.66 | domain still nearly dry — a small `h` loses nothing |
| 3600 | 220.14 | storm on, sheet thickening |
| 5400 | 449.24 | ~220 m³ per 1800 s, linear |
| **7200** | **646.99** | **storm ends** |
| 18000 | 656.19 | inflow hydrographs still running |
| 43200 | 670.04 | +23 m³ over the following 10 h |

The residual accrues at ~220 m³ per output interval **while it rains** and then goes
essentially flat — over the next ten hours, with water still moving, still draining
through the open boundary, and the limiter still firing, it gains 3% of what one
storm interval cost. Continuity is not the leak. Neither is the drift a growth
process: it is roughly fixed in absolute terms once the source stops, which is why
the *relative* error falls as a basin keeps filling.

## 2. The fix

`solver/core/sources.py`. Each cell carries its own float32 **Kahan compensation**
term holding the low-order bits the last source add threw away, which the next add
pays back:

```
y    = rate*dt - comp[i,j]
t    = h[i,j] + y
comp = (t - h[i,j]) - y      # what did not fit
h    = t
```

This is the ledger's own idiom moved onto the grid —
`massbalance._Kahan` has protected the host-side float64 accumulator since M1. `h`
stays **float32**: HANDOFF §2 is untouched, no field is promoted, and the state
grows by exactly one float32 array (2.4 MB at the M6 demo's 590k cells).

Three design calls worth stating:

**Armed only when rain actually falls.** `State.arm_source_compensation()` is called
by `run.py` for a scenario with uniform rain or a rain field. Unarmed, the schemes
launch their original kernels, so every run without an areal source is **bitwise
unchanged** — dam-break, lake-at-rest, the EA benchmarks, M5's `reservoir_release`.

**Point sources are out of scope, deliberately.** Inflow hydrographs measured ~1.3%
of the reach-demo residual (the 9 m³ that accrues between t=7200 and t=18000 with
rain already off). The debt as written is *distributed*-source accumulation, and M7
adds sediment as a second *areal* source. Arming on inflow would also perturb
scenarios carrying no rain at all — `reservoir_release.toml` among them, whose
figures were signed off the same day. If point-source compensation is worth having,
it is a separate change with its own re-measurement.

**The compensated kernel keeps launching at `rain == 0`.** Once the storm stops it
runs with a zero increment, which repays the outstanding compensation into `h`
instead of stranding it. That matters: the outstanding debt is bounded by half an
ulp of `h` per cell, and over a reach-scale grid that bound is itself the order of
the drift being removed.

**Sub-ULP caveat.** The compensation term is exact under Fast2Sum's condition
`|h| >= |y|`. A cell drier than the increment itself (`h < ~1e-4` m in the first
steps of a storm on a dry basin) violates it and its `comp` is then only
approximate — but at that depth the discarded quantity is below `1e-11` m, four
orders below what this exists to remove.

## 3. Validation

`solver/core/test_sources.py`. The gates are **ratios, not thresholds** — a
threshold test passes just as happily if the compensation array is never written, so
it cannot tell a working fix from a fix that was compiled away. Each test runs the
same configuration with and without arming and gates the improvement.

- **Sub-ulp rain** — an increment of `4e-8` m onto `h = 1.0` m is below half an ulp,
  so uncompensated the depth *never moves*: 2000 steps of rain land exactly nothing.
  Compensated it reaches the analytic answer to 0.2%. Error ratio > 100.
- **Fast-math canary** — asserts `h_comp` contains a nonzero entry after a rain
  step. Kahan dies if `(t - h0) - y` is reassociated to zero, and that would make
  every other assertion in the file measure an uncompensated add against itself.
  This is the check that survives a Warp default flip or a future compiler.
- **Realistic storm** — 15 mm/hr onto a 0.3 m sheet at the demo's 20 s steps, the
  reach regime shrunk to a 24×24 CPU grid. Error ratio > 20.
- **Ledger residual** — the gate quantity itself rather than raw depth. Ratio > 20.
- **Spatial rain field, on both schemes** — the field branch is a separate dispatch
  in each scheme and no shipped scenario reaches it (`spatial_fields.toml` has field
  Manning and infiltration but *uniform* rain), so without this test the compensated
  field kernel would first have executed in production. Ratio > 100, and a companion
  test proves the kernel indexes `rain[i,j]` rather than a domain-wide scalar.
- **Unarmed keeps the original kernels** — `h_comp is None` after stepping without
  arming, and the compensated entry points refuse to run unarmed.
- **Arming is idempotent** and does not wipe a running debt mid-run.

CI runs on Warp's CPU backend, so the reach-scale run is not the gate — these
proxies are.

**A trap worth recording, since the field test hit it twice.** These are *sub-ulp*
tests, and that makes them fragile in two specific ways. A rate above half an ulp of
`h` does not round *down* to nothing — it rounds a full ulp *up*, every step, and the
uncompensated control then over-accumulates instead of losing everything. And a
spatially varying rain field on a flat bed tilts `η` and starts the water flowing, at
which point the flux divergence's own float32 round-off at `h = 1 m` is far larger
than the source increment being measured (it swamped the signal by ~9%). Both tests
therefore keep the box at rest and the rate under `5.96e-8`; per-cell indexing is
checked separately, where flow cannot contaminate it.

## 4. Measured result

The failing case, `scenarios/reach_basin.toml` at `coarsen = 4` on an RTX 5090:

**3.77e-06 → 1.28e-07**, a 29× reduction, from over the gate to a seventh of it.
In absolute terms the 647 m³ of storm drift becomes **−3.15 m³**. The ~21 m³ that
remains at t=43200 is flux, limiter and boundary round-off — which this change does
not touch, and which is the floor a source-only fix can reach.

Every rain-bearing recorded figure was re-measured, since they are all stale by
design (the same call M4 made when the conservative donor-β limiter replaced the
non-conservative `max(h, 0)` clamp: the old behaviour was a defect, not a contract):

| scenario | before | after |
|---|---|---|
| in-code demo (M1) | 2.1e-8 | 2.59e-08 |
| `demo_basin_rain` (M2) | 2.12e-8 | 2.59e-08 |
| `--rblaunch` full loop (M2) | 2.12e-8 | 2.59e-08 |
| `river_reach` (M3) | 1.24e-7 | **1.68e-08** |
| `spatial_fields` (M3) | 7.57e-8 | **7.36e-09** |
| `river_reach_hllc` (M4) | 6.66e-7 | **1.31e-07** |
| `reach_basin` (M6), CPU | 2.79e-7 | **7.21e-08** |
| `reach_basin` (M6), CUDA | 3.08e-7 | **1.60e-07** |
| `reach_basin` at `coarsen = 4` | **3.77e-06 — over gate** | **1.28e-07** |
| `reservoir_release` (M5) | 1.36e-7 / 3.15e-7 | unchanged (no areal source) |

**Negative depths are unchanged.** Kahan repays its debt by *subtracting* (`y = -comp`
once the storm stops), so it could in principle push a sharply draining cell below
zero. Measured against the pre-fix commit on `river_reach`, which has open boundaries
and drying cells: **before** `h.min() = -2.33e-09` over 658,496 cells totalling
−0.105 m³; **after** `-2.56e-09` over 652,611 cells totalling −0.105 m³ — the same
population, the same volume, slightly fewer cells. These are the M1 limiter's and
continuity's float32 round-off and pre-date this change; the reach-scale runs
(`reach_basin` at both coarsenings, and the HLLC scenario) report `h.min() == 0.0`
exactly. Nothing here is clamped: `max(h, 0)` invents mass and the repo does not use
it (M4 measured ~6.5e-2 on a drain when it did).

Two honest notes on that table. The M1/M2 demos got **marginally worse**
(2.12e-8 → 2.59e-08): at that magnitude source drift was never what set the
residual, and compensation slightly perturbs the trajectory that does. And the M6
sign-off's claim that `--rblaunch` lands "2.12e-08, bit-for-bit M2's figure" is
**broken by design** — the loop is still green end to end (`starting → running →
writing → done`, 13 frames, no Windows file-handoff race), at 2.59e-08.

234 tests green, ruff clean.

## 5. What this does *not* fix

- **Flux-divergence round-off.** Untouched, and now the floor: ~21 m³ on the
  reach-demo case. Reaching below it means compensating continuity, which is a much
  larger change to the hot kernel of both schemes and is not justified by anything
  measured yet.
- **Point sources.** See §2.
- **The `η = h + z` conditioning problem.** A different float32 hazard with a
  different fix (`[grid] datum`, M5) — a centimetre sheet on a bed at altitude loses
  `h` inside the sum. Compensation does not help there and the datum shift is still
  the answer.
- **The reach-scale diagnosis discipline.** Still check storm depth, cell count and
  step count before calling a gate exceedance a bug. The envelope is much wider now,
  not infinite.
