# Point-source compensation — the inflow add, measured

**Status: done, 2026-08-17.** The second of M7's carried findings
(`roadmap.md` → "Carried out of M7", item 2), raised by the scheduler pass and left
for its own commit because it needs its own before/after across every
inflow-bearing scenario. Not a milestone: a scoped correctness pass on one
arithmetic path, and — more usefully — a **measurement that corrects the finding
that asked for it**.

---

## 1. The defect, and what was actually true about it

`[[inflow]]` injects a discharge as a cell source: each step adds `Q(t)·dt` cubic
metres to a named cell, i.e.

```
h[i,j] = h[i,j] + Q*dt/area        # float32, once per source cell per step
```

That is the same float32 accumulator the precision pass fixed for rain, at a
different arity. The precision pass (`precision-sources.md` §2) deliberately scoped
it out, on the measurement that inflow was **~1.3 % of the reach-demo residual** —
and that measurement was taken on a **rain-driven** run, where a couple of hundred
thousand raining cells swamp four injecting ones. The scheduler pass found it does
not transfer: on `reach_alluvial`, which is flood-driven with no rain at all,
`outflow_cum` is exactly 0.0 for the whole run, so every cubic metre of residual is
stored float32 volume weighed against the float64 inflow ledger.

### 1.1 What the carried item claimed, and what the measurement says

The carried item's words were that "the entire mass residual is stored float32
volume against the float64 inflow ledger", and that "the likely seat is the four
`[[inflow]]` cells". The first half is true. The second half is **partly true, and
now it is measured rather than inferred.**

The instrument: wrap the injector so the target cells are read in float64
immediately before and after the launch (a four-element gather, so no whole-grid
transfer), and accumulate the difference against the volume the injector reported to
the ledger. That isolates the source add from flux divergence, the limiter,
morphology and everything else in the step. On `reach_alluvial`, 24 h at 100 m on
the 5090, 5630 steps, four inflow cells:

| | requested (m³) | applied to the field (m³) | applied − requested |
|---|---|---|---|
| uncompensated | 4 445 999.972597 | 4 446 001.187619 | **+1.215022 m³** |
| compensated | 4 445 999.972597 | 4 445 999.972504 | **−0.000093 m³** |

The point source was putting **1.2 m³ more water into the field than the ledger was
told about**, and compensation removes 99.99 % of that — a **13 000×** reduction in
magnitude, and with the residue landing on the other side of zero, which is what a
Kahan term that is merely carrying an unrepresentable remainder looks like rather
than one still losing bits.

The run's *total* residual is **2.13 m³** in magnitude (4.79e-07 of a 4.45 million m³
peak), and it falls to **1.18 m³** (2.66e-07) with the add compensated. So removing a
1.2 m³ term moved the total by 0.95 m³.

**Resisting the tempting arithmetic:** that is *not* "the inflow add was 57 % of the
residual". The attribution above is **signed** — the add pushed stored volume *above*
what was banked — while `scheduler-equal-steps.md` §8.5's ledger row shows the total
residual as **−2.119 m³**, stored volume *below* the ledger. Opposite signs, so the
two do not subtract, and the 0.95 m³ improvement is consistent with partial
cancellation rather than with a clean split. What is measured and defensible is the
pair of statements above: **the add's own error fell 13 000×, and the run's total
residual roughly halved.** Whatever remains is flux-divergence and limiter round-off —
the floor `precision-sources.md` §5 already named as untouched, and still untouched
here.

### 1.2 Why it is a drift at all, and why it stops being one

Rounding to nearest has zero mean, so the obvious expectation is a random walk, not
a drift. What makes it a drift is **correlation**: while nothing else writes `h`,
the low-order bits of `h` and of the increment are the same every step, so the same
rounding decision is taken over and over and the errors add rather than cancel.

That is measurable, and it is the single most useful thing this pass learned,
because it explains both the size of the fix and the shape of its tests:

| fixture | what writes `h` | uncompensated vs compensated |
|---|---|---|
| pure accumulation, 1500 steps | the source only | **489×** |
| pure accumulation, 4000 steps | the source only | **563×** |
| walled pond + `step()`, 300 / 600 / 1200 steps | source + continuity | 0.8× / 2.8× / 7.0× |

Once continuity rewrites `h` every step the low bits decorrelate, the systematic
drift becomes a random walk, and an A/B on a flowing fixture's mass residual comes
out as **noise — sometimes "worse"**. That is not evidence the fix does nothing; the
attribution table in §1.1 measures the same flowing run and shows the add's own
error falling 13 000×. It is evidence that *the ledger residual of a short flowing
fixture is the wrong instrument*, and it is why §3's gates are written on
accumulation rather than on a stepped run's mass balance.

## 2. The fix

`solver/processes/inflow.py`. Each **source entry** carries its own float32 Kahan
compensation term, through `solver.core.sources.kahan_add` — the shared `wp.func`
that module's docstring exists to insist on, so a float32 accumulator has one
definition of "how" wherever it appears (rain, M7's transport integral, now this).

```
y    = Q*dt/area - comp[k]
t    = h[i,j] + y
comp[k] = (t - h[i,j]) - y      # what did not fit
h[i,j] = t
```

Four design calls worth stating.

**The compensation is indexed by source entry, not by cell.** That is what keeps one
thread per entry safe: the constructor already rejects duplicate cells (two
non-atomic adds on one cell would race and break determinism), so no two threads
touch the same `h` or the same `comp` slot. It also means the array is a handful of
floats regardless of grid size.

**It is owned by the injector, not armed on `State.h_comp`.** Both schemes dispatch
their *areal* source kernels on `h_comp is not None`
(`local_inertial.py`, `hllc.py`), so arming it for a point source would drag every
rain-free scenario onto a different code path — and would have forced retiring
`test_sources.py::test_unarmed_state_keeps_the_original_kernels`. A separate array
costs nothing and leaves both schemes untouched.

**No arming condition.** Rain compensation is armed only when rain actually falls,
because the array is grid-sized and the schemes branch on it. Here the injector only
exists when a scenario declares `[[inflow]]` at all, so "armed" and "has a point
source" are already the same condition. `compensated=False` exists to reproduce the
pre-fix arithmetic for the A/B gates; it is not a production switch, and the
uncompensated kernel is kept solely as that control arm.

**The ledger still banks the float32 *request*.** `apply()` returns
`sum(add_depth) · area` exactly as before — deliberately **not** the field's actual
delta. Banking the delta would make the residual read ~0 by construction and blind
the mass gate to this entire defect class. The ledger is the independent witness;
compensation makes the field catch up to it, not the reverse. This is the same trap
the scheduler pass spent a commit documenting in a different guise: a gate
calibrated by its own measurement is not a gate.

**Sub-ULP caveat**, inherited from `sources.py`: the term is exact under Fast2Sum's
`|h| ≥ |increment|`. The first injection onto a dry cell violates that — but a dry
cell is also where the add loses nothing, so what the term fails to capture is below
what it exists to remove.

## 3. Validation

`solver/processes/test_inflow.py`, five new tests, **ratios not thresholds** (a
threshold passes just as happily if the compensation array is never written).

- **Realistic flood increment** — `reach_alluvial`'s regime shrunk to a unit test:
  90 m³/s into a 100 m cell is 0.027 m of float32 onto metre-deep water, 1500 steps.
  Uncompensated the field ends ~5 m³ from the 405 000 m³ the ledger banked;
  compensated ~0.01 m³. Ratio > 100 (measured 489).
- **Sub-ulp point source** — 4e-8 m onto 1.0 m is under half an ulp, so 2000
  uncompensated adds land *exactly nothing* while the ledger banks the full volume;
  the test asserts the error **equals** the requested volume, then gates the ratio
  at > 100. This is the sharpest statement of the defect.
- **Fast-math canary** — the compensation array must contain a nonzero entry after
  one add. If `(t − h) − y` is ever reassociated to zero, every other assertion in
  the file would be measuring an uncompensated add against itself.
- **The control arm is the plain add** — `compensated=False` checked bit-for-bit
  against a hand-rolled float32 running sum. A control is only a control if it is
  the arithmetic the fix replaced.
- **A receded hydrograph keeps carrying its debt** — rise, peak, recede to zero,
  then a 5100 s dry tail. There is no zero-discharge shortcut in `apply()`, so the
  kernel keeps launching and the debt is repaid the moment it is representable
  rather than stranded. Ratio > 20 at the end of the tail, and the debt is asserted
  bounded (it is low-order bits of `h`, not a growing quantity).

Deliberately **not** gated: the mass residual of a stepped, flowing fixture. §1.2
measures that A/B at 0.8× / 2.8× / 7.0× — a coin flip. Gating it would have meant
choosing the step count that gave the answer wanted, which is the failure mode the
scheduler pass's ripple gate was written to avoid.

**358 tests green** (353 before), ruff clean.

## 4. Measured result

Every scenario carrying `[[inflow]]`, each run twice with only the injector's
compensation swapped. Scenarios with no `[[inflow]]` construct no injector at all
(`run.py` builds it only `if scenario.inflows`), so `demo_basin_rain` and
`spatial_fields` are **bitwise unchanged by construction**, not by measurement.

| scenario | backend | uncompensated | compensated |
|---|---|---|---|
| `river_reach` (M3) | CUDA | 1.84e-08 | 1.80e-08 |
| `river_reach_hllc` (M4) | CUDA | 1.51e-07 | **1.16e-07** |
| `reservoir_release` (M5) | CUDA | 2.38e-07 | 2.32e-07 |
| `reservoir_release` (M5) | CPU | 1.30e-07 | *1.87e-07* |
| `reach_basin` (M6) | CUDA | 5.95e-08 | *6.00e-08* |
| `reach_basin` (M6) | CPU | 6.07e-08 | *6.10e-08* |
| `reach_alluvial` (M7) | CUDA | 4.79e-07 | **2.66e-07** |
| `reach_alluvial` (M7) | CPU | 5.45e-07 | **2.14e-07** |

**Every "uncompensated" arm reproduces its recorded *mass* figure exactly** — 1.84e-08,
1.51e-07, 2.38e-07, 1.30e-07, 5.95e-08, 6.07e-08, 4.79e-07, 5.45e-07 are the numbers
`CLAUDE.md` carried from the scheduler pass, to the digit. Nothing else in the tree or the data drifted,
so the deltas are the injector's and only the injector's.

One number in this batch looks like an exception and is not. `reach_alluvial`'s
uncompensated **bed volume** on CPU reads 168 911 m³ against the 168 563 m³ `CLAUDE.md`
carried — 2.1e-3 apart, twenty times looser than anything in the table above. That
recorded figure is the **CUDA** one: it comes from `scheduler-equal-steps.md` §8.5,
whose A/B table pairs it with the 4.79e-07 mass residual that `CLAUDE.md` labels CUDA,
and the two backends only ever agreed to ~1e-4 in bed volume against ~1e-9 in mass.
Both are labelled with their backend now. The reproduces-exactly claim is about the
mass column, which is the column this pass moves.

### 4.1 Two scenarios got marginally worse, and that is the noise, not a regression

`reach_basin` moves 5.95e-08 → 6.00e-08 on CUDA (+0.8 %) and 6.07e-08 → 6.10e-08 on
CPU (+0.5 %), and `reservoir_release` on CPU moves 1.30e-07 → 1.87e-07 (+44 %). All are
the **total** residual, which on those runs is dominated by a term this change does not
touch, and compensation perturbs the trajectory that sets it.

The cheapest evidence that it is noise needs no extra run at all: `reservoir_release`
moves **down** on CUDA and **up** on CPU. Both backends execute the *same* deterministic
`Δt` sequence, so a sign flip between them under an otherwise irrelevant change is
what noise looks like.

The confirming experiment, and **the reading was written down before the numbers came
back** (the scheduler pass's own lesson — a criterion chosen after seeing the answer is
not a criterion): re-run `reservoir_release` on CPU at `dt_max` 9.0 and 8.0, which
genuinely re-partitions the step sequence. *If all three partitions move up by a similar
factor, that is a systematic cost and gets named as a carried caveat; if they do not all
move the same way as the shipped one, it is noise.* Result:

| `dt_max` | uncompensated | compensated | factor |
|---|---|---|---|
| 10.0 (shipped) | 1.30e-07 | 1.87e-07 | 1.44× |
| 9.0 | 1.74e-07 | 1.68e-07 | **0.97×** |
| 8.0 | 1.32e-07 | 3.33e-07 | 2.52× |

The three partitions genuinely differ (the uncompensated arm alone spans
1.30–1.74e-07), so the perturbation did what it was meant to. **One of the three moves
the other way, so by the criterion above this is noise** — and the shape of the other
two says the same thing independently: a systematic cost shows a *consistent* factor,
the way `scheduler-equal-steps.md` §8.5's genuinely systematic ~9× did across its three
partitions. 0.97× / 1.44× / 2.52× is not that. Every value stays 3–7.5× under the gate.

`reach_alluvial` is the case where the systematic term is large enough to dominate the
noise, which is exactly why it was the scenario that surfaced the finding.

## 5. What this does *not* fix

- **The rest of `reach_alluvial`'s residual.** Roughly half of it survives (2.66e-07,
  ~1.2 m³), and it is flux-divergence and limiter round-off over 590 k cells and
  5630 steps. Reaching below that means compensating continuity inside the hot
  kernel of both schemes, which nothing measured yet justifies —
  `precision-sources.md` §5 said so and it is still true.
- **Reservoir release delivery.** `solver/processes/reservoir.py` banks the *actual*
  float32 depth change in float64 rather than compensating the add — a different and
  internally consistent strategy for a transfer that must be mass-exact between two
  places in the same domain. Not examined here, and not obviously wanting the same
  treatment.
- **The infiltration sink.** An areal *sink* on the same fields, still a bare
  float32 subtract with its loss banked in float64. It has the same shape as the
  defect above and has never been measured; it is not in this pass's scope, and no
  scenario has pointed at it.
- **`h` staying float32.** HANDOFF §2 is untouched: no field is promoted, and the
  state grows by a few floats.
