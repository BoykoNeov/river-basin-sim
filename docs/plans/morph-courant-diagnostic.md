# The morphological Courant diagnostic — make the number mean what it says

The last finding carried out of M7 (`roadmap.md`, "Carried out of M7" item 3). The
warning that says *"the bed wave crosses more than a cell per activation, so the bed
change is a splitting artefact"* fires on the M7 demo at **46 425** — and the bed is
right to **0.9%** when the interval is halved. A diagnostic that is wrong by four
orders of magnitude on the only shipped scenario that arms it does not warn anybody;
it trains them to ignore the line.

This is a **diagnostic-only** change. Nothing here reads into the physics: the bed,
the water, the ledgers and the store's `bed_change` field must come out **bit-for-bit
identical** on every scenario. That is both the design constraint and the cheapest
possible verification (§4).

---

## 1. What is actually wrong, in two parts — and the carried finding names only one

### 1.1 The reduction picks a cell the law does not apply to

`morphology.advance` reports `celerity_field(state).max()`. `celerity_field` evaluates
`bed_celerity` per cell, and `bed_celerity ∝ (θ/h)·√(θ−θ_c)` with `θ ∝ n²q²/h^{10/3}` —
so as a cell approaches the wet/dry guard the celerity **diverges**. Every shipped
scenario starts dry and advances a wetting front, so there is always such a cell.

Measured on the M7 demo (plan `M7-morphology.md` step 9): of 1414 cells with nonzero
celerity, **exactly one** sits below `h_col/d50 = 10`, and that one cell sets the
46 425. The peak over cells the law applies to (`h_col/d50 ≥ 35`) is **19.4**.

A max over the whole field is therefore reporting the guard, not the reach.

### 1.2 …and 19.4 is still an overstatement, which the carried finding does not say

This is the part to be careful about, and it is why this cannot be a one-line filter.
**19.4 is also over the gate of 1.** The in-regime peak runs steadily **8–16** across
the run. So on the demo, the splitting is "unresolved" by this diagnostic's own
definition *everywhere the transport is real* — and the bed is nonetheless right to
0.9% in volume, ~1 mm in peak scour/fill, and 0.9992 correlated on channel cells when
`interval_s` is halved from 900 s to 450 s.

**So a filter alone will probably not silence the demo, and the plan must not promise
that it will.** Whether it does is a measurement (§3, step 2), not an assumption.

The likely reason the threshold itself does not transfer is written down already, in
`validation/bedwave.py`'s "What the analytic reference assumes":

> `bed_celerity` linearises Exner at **fixed unit discharge and a rigid water
> surface** … Reality sits between that and the opposite limit, where the surface
> follows the bed so closely that the depth never changes and the bed wave has no
> celerity at all. Which limit a fixture is in is set by `sigma/(h/S)` …

The celerity fixture **enforces** that limit — constraint (2), `bump_slenderness <
0.05`, asserted in `test_the_fixture_is_sized_for_the_law_not_for_convenience`. A
real basin does not honour it. So `c_b` is a **rigid-lid upper bound** on a reach
whose bed features are not slender bumps, and a Courant number built from an upper
bound inherits the overstatement. Where the assumption holds, the diagnostic is
sharp — the Courant-3.30 fixture is the proof, and §5 keeps it.

**The honest summary of the defect is therefore: the number is a one-sided bound, it
is reduced in the way most likely to maximise it, and it is presented as a
measurement.** Part 1.1 is fixable arithmetic. Part 1.2 is a property of the
reference and is fixed by *saying so*, not by tuning a constant.

---

## 2. The design

### 2.1 What is added, and what is deliberately not touched

**`courant` keeps its exact present meaning.** Three assertions read it straight out
of the record or `.zattrs`, one of them a *lower* bound
(`test_bed_wave.py` `0.5*fx.courant < res.courant < 2.0*fx.courant`;
`test_run.py` `> 1.0` and `== 0.0`). Redefining the key moves at least three gates
and makes every before/after ambiguous. **Companion keys move none.**

Three new fields on `BedChangeRecord` (and therefore on `.zattrs` `morphology`,
which is additive by §7.2):

| field | what it is |
|---|---|
| `courant_moving` | peak Courant over cells that carried bed change **this activation** |
| `over_courant_share` | fraction of this activation's gross \|Δz\| sitting in cells with Courant > `MORPH_COURANT_GATE` |
| `courant_cells` | how many cells are over the gate (so "one cell" is visible as one cell) |

**No relative-submergence cut-off.** `h ≥ k·d50` was refused at M7 build step 8 for a
reason that still holds: `_sediment_bowl` runs at `h/d50 = 0.5`, so a submergence
guard would keep several tests green *for a second, unrelated reason*
(`solver/test_run.py:319-333` says this at length, and
`validation/test_morphology_gates.py:161` repeats it). Weighting by **where the bed
actually moved** achieves the same discrimination with no new physical constant: a
cell that is dry-ish and fast but moves no bed drops out on its own weight.

### 2.2 Why "where the bed moved" is the right weight

The splitting error is the bed change applied at once; a cell that applies none
cannot contribute one, whatever its celerity. And a genuinely unresolved bed wave
**does** move bed — the Courant-3.30 fixture's bump grows to 1.63× its height — so
this weighting cannot hide the failure mode the diagnostic exists for. It is also
exactly what the warning text already tells the reader to do by hand
(`run.py:469`, *"check where the bed change actually is before acting"*).

### 2.3 Cost

`advance()` already pulls `dz_cum` to the host every activation. Keeping the previous
copy and differencing gives the per-activation field for free; `celerity_field` is
already a host array. **No new device→host transfer, no new kernel**, once per
activation (900 s of simulated time by default).

**One trap, named because it fails in the direction of the hoped-for answer.** The
retained field must be a `.copy()`, not the array `dz_cum.numpy()` hands back. An
alias overwritten by the next activation's fetch differences to **zero everywhere**,
which reports `over_courant_share = 0.0` — the reading that would let this pass
declare success on a bug. Gated directly: for an activation known to move the bed,
the differenced field must sum to `applied_m3 / (A·(1−p))`, which is already computed
beside it.

### 2.4 The trigger

**Asked and answered — with a caveat that outranks the answer.** The user chose
"quiet on a lone cell, plus the numbers". That question was posed on the premise that
the peak is a lone guard cell, and §1.2 (written after the question was asked)
says it is not: the in-regime peak is 8–16 across the whole run, so the demo is over
the gate *where the transport is real*. **A filter of any kind therefore probably
cannot reach "quiet"** — only moving `MORPH_COURANT_GATE` could, and §5 refuses that.
The choice is recorded as the intended branch and the prediction against it is
recorded beside it; the measurement in §3 step 1 settles which is right, and the user
gets told before any code is written against either.

If the quiet branch survives the measurement, the firing condition moves from

```
peak_courant >= MORPH_COURANT_GATE
```

to a share-based rule, and the message leads with the share and the cell count rather
than the raw peak. `MORPH_COURANT_GATE = 1.0` is **unchanged** — the share is what is
new, not the threshold — so nothing recalibrates the physics claim.

Two consequences, both real work and both in the build order:

1. **`test_a_scenario_over_the_morphological_courant_gate_warns` must be re-homed.**
   It is built on `_sediment_bowl`, which transports at `h/d50 = 0.5` — a sheet
   shallower than one grain. Its own docstring prescribes the remedy verbatim:
   *"re-home them to channel flow then, do not weaken the assertions."* The
   replacement is a small sub-grid-channel reach (M6 machinery, already available in
   the test suite) driven over-Courant on purpose.
2. **The share threshold is a new constant and must be justified, not picked — and
   the prediction is that it cannot be.** The discriminating quantity is named here,
   before it is measured, so the measurement can refute it: **the share of
   `reach_alluvial`'s gross \|Δz\| sitting in cells over the gate.** §1.2 predicts
   that share is **large** — not set by the lone guard cell at all, but by the 8–16
   in-regime cells, which is where most of the bed change is. If that prediction
   holds, **no choice of threshold reaches "quiet"**, because the demo is over the
   gate exactly where the transport is real; the only thing that would silence it is
   moving `MORPH_COURANT_GATE` itself, which §5 refuses. In that case the trigger
   does **not** move, the new keys and the message change still land, and §1.2 becomes
   the finding this pass carries. Calibrating the constant until the demo goes quiet
   is the failure mode to avoid — it is the mistake `test_clamp_ripple.py` was built
   to prevent during the scheduler pass (*"a gate calibrated by its measurement
   window"*).

---

## 3. Build order

1. **Measure before touching anything.** Instrument a throwaway script (not the
   repo) that re-runs `reach_alluvial` and the celerity fixture and dumps, per
   activation: raw peak Courant, the peak restricted to cells that moved, the
   over-gate share, the cell count, and the submergence distribution. Record the
   numbers. This produces the "before" column and settles §1.2.
   - **Also settle the roadmap's stated blocker.** It claims a regime-aware
     diagnostic *"would silently change what the Courant-3.30 fixture asserts."* The
     fixture runs at `h ≈ 1.5 m` on `d50 = 8 mm` — `h/d50 ≈ 187`. Confirm the
     distribution, and if it is as expected, **the stated reason for the deferral was
     wrong** and the plan says so (the real blocker is `_sediment_bowl`, which the
     roadmap never names).
2. **Decision point.** With step 1's numbers: does the demo's bed change actually
   concentrate in in-gate cells? If yes → move the trigger onto the share (the user's
   choice). If no → the trigger stays where it is, the new keys and the message
   change still land, and §1.2 becomes the finding this pass carries. **Write down
   which branch was taken and why.**
3. **`solver/core/sediment.py`** — a small host helper beside `celerity_field` that
   takes the celerity field and the activation's `Δz` field and returns the three
   summary numbers. Pure numpy, no kernel, nothing in the physics reads it. It sits
   here rather than in `morphology.py` because `celerity_field` and
   `MORPH_COURANT_GATE` already do, even though its `Δz` argument is a
   morphology-process concept — **the whole summary lives in one file either way**,
   it is not split across the two.
4. **`solver/processes/morphology.py`** — hold the previous `dz_cum` copy, difference
   it, populate the new record fields, keep `courant` byte-identical. Track the
   companion peaks the way `_peak_courant` is tracked.
5. **`solver/run.py`** — the warning: trigger per step 2, message leads with share +
   cell count, still names `interval_s`, still prints unconditionally. Keep the
   existing "check the bed against a longer interval" remedy — step 1 will have
   re-confirmed it.
6. **Tests.** Unit tests for the helper (a synthetic field with one hot cell that
   moves nothing, and one with the bed change *in* the hot cells — the pair is the
   argument, one arm alone passes by construction). The re-homed scenario-level
   warning test if step 2 took the quiet branch. A test that the demo-shaped case —
   one guard cell at a huge celerity — reports a share near zero.
7. **The before/after run** (§4) and the docs.

---

## 4. Validation

**The keystone is that nothing moved.** This is a pure observer, so:

- `reach_alluvial` on CUDA and CPU: `bed_change` field **bit-for-bit identical** to
  the before run, gross volume identical, water mass residual identical, sediment
  residual identical. Not "within tolerance" — *identical*. If it is not, the change
  leaked into the physics and that is the whole finding.
- The full suite green with **no gate's tolerance touched**. The scheduler pass set
  the standard here: run the changed code against the untouched suite *first*, and
  expect the only failures to be the tests this plan says are being re-homed. Any
  other failure is a real one.
- `test_an_interval_that_moves_the_bed_a_cell_per_activation_is_caught` still asserts
  exactly what it asserted, on an unmoved `courant` value. This is the fixture that
  proves the Courant gate and the celerity gate are not substitutes — it is the one
  thing this pass is most likely to damage silently.

**The numbers table** (the deliverable of the pass): for `reach_alluvial`, raw peak /
moving peak / over-gate share / cells over gate, before and after, alongside the
already-measured 0.9% interval-halving sensitivity. Plus the same row for the
re-homed over-Courant scenario, which must be loud on all four.

---

## 5. What this does *not* fix

- **The understatement stays.** `celerity_field` samples at the activation *instant*,
  so an interval in which a flood arrived, moved 577 m³ of bed and drained away
  reports Courant **0.000** — pinned by
  `test_the_courant_diagnostic_samples_the_flow_and_a_drained_run_reads_zero`, whose
  assertion is `== 0.0`. Making it a true interval maximum costs a full-field host
  reduction **every fast step**, which M7 build step 8 priced and refused for a
  diagnostic. Folding it in here would also make the before/after unreadable — the
  bed-weighted share and an interval maximum would move the same number for two
  different reasons. **Still carried, deliberately.**
- **`c_b` stays a rigid-lid upper bound.** §1.2's mechanism is not fixed, it is
  documented. A slenderness-aware celerity is a change to the *reference* the
  celerity gate is measured against, which is a physics change with its own
  validation, not a diagnostic pass.
- **`MORPH_COURANT_GATE` stays at 1.0.** The Courant-3.30 fixture is direct evidence
  that 3.3 is already broken *where the reference is valid*. Raising the constant to
  make the demo quiet would be calibrating a threshold by the answer it gives on one
  scenario — the opposite of what this pass is for.
- **No relative-submergence guard anywhere**, for the reason in §2.1. If a future
  pass wants one, `_sediment_bowl`'s docstring already states the contingency: the
  tests built on it must be re-homed to channel flow, not weakened.
