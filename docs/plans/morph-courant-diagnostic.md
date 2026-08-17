# The morphological Courant diagnostic — make the number mean what it says

The last finding carried out of M7 (`roadmap.md`, "Carried out of M7" item 3). The
warning that says *"the bed wave crosses more than a cell per activation, so the bed
change is a splitting artefact"* fires on the M7 demo at **39 271** — and the bed is
right to **0.9%** when the interval is halved. A diagnostic that is wrong by four
orders of magnitude on the only shipped scenario that arms it does not warn anybody;
it trains them to ignore the line.

> **The headline number moved between M7 and this pass: 46 425 → 39 271.** Same
> scenario, same `interval_s`; the scheduler equal-steps pass and point-source
> compensation both changed the Δt sequence, and the peak is a field maximum over a
> wetting front, which is the least reproducible statistic in the run. Quote the new
> figure. This is exactly the "unlabelled number invites a false *did not reproduce*"
> hazard `CLAUDE.md` already flags for bed volumes.

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

> **§3 step 1 measured this and half of it is wrong.** The in-regime peak reproduces
> exactly (**19.38**), and up to **14** cells sit below `h_col/d50 = 10` at once rather
> than one. But the guard cell **moves bed**: over the whole run the peak restricted to
> cells that carried bed change this activation is **39 271.12** — *the same number, to
> every digit*. See §1.3.

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

### 1.3 What §3 step 1 actually measured

Four runs through one observer wrapper on `MorphologyProcess.advance`, which
differences `dz_cum` per activation and evaluates `celerity_field` beside it. All
figures below are that observer's; nothing in the solver moved to obtain them.

| | fixture 45 s | fixture 900 s | `reach_alluvial` | miniature + rain sheet |
|---|---|---|---|---|
| activations | 103 | 5 | 96 | 12 |
| peak Courant, raw | 0.162 | 3.298 | **39 271** | 5.0e8 |
| … over cells that moved bed | 0.162 | 3.298 | **39 271** | 5.0e8 |
| … over cells at `h_col/d50 ≥ 35` | 0.162 | 3.298 | **19.38** | 88.3 |
| max over-gate share of \|Δz\| | 0.000 | 1.000 | 0.883 | 0.477 |
| gross-weighted over-gate share | 0.000 | 1.000 | **0.495** | **0.012** |
| cells over the gate (max) | 0 | 239 / 240 | 578 / 1414 | 48 |
| cells below `h_col/d50 = 10` | 0 | 0 | 14 | 23 |

`reach_alluvial` reproduced its recorded run while being observed — mass **2.66e-07**,
sediment **4.21e-17**, bed **168 557 m³** against the recorded 168 563 m³ CUDA — and
the aliasing gate of §2.3 held: differenced field vs the record's own `applied_m3`,
max absolute error **1.4e-12 m³**.

**Three findings, in descending order of how much they change the plan.**

1. **A share-based trigger is not merely unable to go quiet — it is anti-correlated
   with badness, and that closes the branch on principle.** The last column is the
   rain-sheet arm of `validation/test_morphology_gates.py::_regime_share`: the
   configuration this repo *removed rain from `reach_alluvial` to avoid*, which moved
   **1.9e9 m³** of nonsense bed and, as a scenario bug, once read `bed_moved` 1.46e11.
   Its share is **0.012** — an order of magnitude *below* the demo's 0.495, and below
   the demo's max-share too. **Under either aggregation, every threshold that quiets
   the demo also quiets the rain sheet.** The mechanism is §5's understatement
   compounding with the weighting: the sheet's bed moved where the flow had already
   left, so those cells read celerity zero at the sampling instant and carry the
   volume anyway.
2. **§2.2's premise is refuted.** Weighting the *peak* by where the bed moved changes
   nothing on the run that matters: 39 271.12 either way. Per activation the filter
   does bite (at t = 1800 s, 1016.5 raw against 16.7 moving), but the guard cell moves
   bed often enough that the run maximum is untouched. `courant_moving` therefore
   ships as an *honest null* — recorded because a reader comparing it to `courant`
   learns something real, not because it fixed §1.1.
3. **The roadmap's stated blocker is false, and cleanly so.** It claims a regime-aware
   diagnostic *"would silently change what the Courant-3.30 fixture asserts."* The
   fixture runs at `h/d50 = 187.0`, its **minimum** submergence over live cells is
   **130.1**, it has **zero** cells below `h_col/d50 = 10`, and its raw, bed-weighted
   and in-regime peaks are *numerically identical* on both arms. No plausible cut-off
   touches it. The real blocker is `_sediment_bowl` at `h/d50 = 0.5`, which the
   roadmap never names — and §2.1 named it before the measurement.

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
| `courant_in_regime` | peak Courant over cells at `h_col/d50 ≥ MORPH_REGIME_FLOOR` |
| `over_courant_share` | fraction of this activation's gross \|Δz\| sitting in cells with Courant > `MORPH_COURANT_GATE` |
| `courant_cells` | how many cells are over the gate (so "one cell" is visible as one cell) |

**A relative-submergence cut-off, but only ever as a number that is printed.** The
plan opened refusing one, because `h ≥ k·d50` was refused at M7 build step 8 for a
reason that still holds: `_sediment_bowl` runs at `h/d50 = 0.5`, so a submergence
guard would keep several tests green *for a second, unrelated reason*
(`solver/test_run.py:319-333` says this at length, and
`validation/test_morphology_gates.py:161` repeats it).

**That objection is about a cut-off in a trigger or a gate, and §1.3 froze the
trigger.** With nothing acting on it, `courant_in_regime` changes no test's outcome
and cannot hold one green — and it is the most informative number in the whole
dataset, 19.38 against 39 271, a **2000×** contrast, and the figure the carried
finding already quotes. So `_REGIME_FLOOR = 35.0` moves out of
`validation/test_morphology_gates.py` and into `solver/core/sediment.py` as
`MORPH_REGIME_FLOOR`, with the validation module importing it rather than keeping its
own copy. The cost is stated rather than hidden: **a regime constant now lives in the
solver**, for a diagnostic. If it ever becomes load-bearing, M7 step 8's objection
comes straight back and `_sediment_bowl`'s docstring already prescribes the remedy.

**Weighting by where the bed moved stays, and stays honest about what it did.** §1.3
finding 2 measured that it does not drop the guard cell from the run maximum. It is
recorded because the comparison is informative per activation, not because it fixed
§1.1.

### 2.2 Why "where the bed moved" is the right weight — and where it is not

The splitting error is the bed change applied at once; a cell that applies none
cannot contribute one, whatever its celerity. And a genuinely unresolved bed wave
**does** move bed — the Courant-3.30 fixture's bump grows to 1.63× its height — so
this weighting cannot hide the failure mode the diagnostic exists for. It is also
exactly what the warning text already tells the reader to do by hand
(`run.py:469`, *"check where the bed change actually is before acting"*).

**Both halves of that argument survive as a description and fail as a trigger**, and
§1.3 is why. The converse of "a cell that moves no bed contributes no error" is not
true in the form the trigger would need: a cell can move a great deal of bed and read
celerity **zero**, because `celerity_field` samples the flow at the activation instant
and the flood that moved the bed has already passed (§5). The rain-sheet arm is that
case at full size — 1.9e9 m³ moved, share 0.012. So the weighting is a good way to
*describe* an activation and a dangerous way to *decide* whether to warn.

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

### 2.4 The trigger — asked, answered, and then refuted by the measurement

**The user chose "quiet on a lone cell, plus the numbers".** That question was posed
on the premise that the peak is a lone guard cell. §1.2 doubted the premise before
the measurement; §1.3 destroyed it. **The trigger does not move.** The firing
condition stays exactly

```
peak_courant >= MORPH_COURANT_GATE
```

and `MORPH_COURANT_GATE` stays 1.0. What lands is the second half of the user's
answer — *plus the numbers* — in full.

**Why, in one line each.** A share-based trigger cannot go quiet on the demo (0.495
gross-weighted, 0.883 peak); a bed-weighted peak does not move the demo's headline at
all (39 271 either way); and — decisively — **any threshold that quiets the demo also
quiets the rain sheet**, which is the one configuration in the repo with no other
symptom. Calibrating a constant until the demo goes quiet was already the named
failure mode here, the mistake `test_clamp_ripple.py` was built to prevent during the
scheduler pass (*"a gate calibrated by its measurement window"*). The measurement
found something worse than an uncalibratable constant: a statistic that points the
wrong way.

**Two items this deletes from the build order rather than completing.**

1. **`test_a_scenario_over_the_morphological_courant_gate_warns` is not re-homed.**
   The re-homing existed only to stop a moved trigger from silencing a
   `h/d50 = 0.5` fixture. The trigger did not move, so the test stays green on
   `_sediment_bowl`, untouched. `_sediment_bowl`'s standing contingency is unspent.
2. **There is no share threshold**, so there is no new constant to justify.

**What the reader gets instead.** The warning keeps firing on the same condition and
stops pretending the peak is a measurement of the reach: it leads with the in-regime
peak and the over-gate cell count against the live cell count, and prints the raw
peak as what it is. A reader who sees *"39 271 peak, but 19.4 over the 1400 cells the
law applies to, and 578 of them over the gate"* can act; one who sees 39 271 alone
learns to skip the line.

---

## 3. Build order

1. ~~**Measure before touching anything.**~~ **Done** — §1.3. A throwaway observer
   (kept out of the repo, `M:\claud_projects\temp\morph-courant\measure.py`) wrapped
   `MorphologyProcess.advance` across four runs. It also settled the roadmap's stated
   blocker: **wrong**, and §1.3 finding 3 says why.
2. ~~**Decision point.**~~ **Done — the trigger stays.** §2.4 records the branch and
   the reason. The user's choice was refuted by the measurement it was scheduled to
   be tested against, which is what the step existed for.
3. **`solver/core/sediment.py`** — a small host helper beside `celerity_field` that
   takes the celerity field, the activation's `Δz` field, the interval and `dx`, and
   returns the summary numbers. Pure numpy, no kernel, nothing in the physics reads
   it. It sits here rather than in `morphology.py` because `celerity_field` and
   `MORPH_COURANT_GATE` already do, even though its `Δz` argument is a
   morphology-process concept — **the whole summary lives in one file either way**,
   it is not split across the two. `MORPH_REGIME_FLOOR` lands here too, and
   `validation/test_morphology_gates.py` imports it instead of keeping its own copy.
4. **`solver/processes/morphology.py`** — hold the previous `dz_cum` copy, difference
   it, populate the new record fields, keep `courant` byte-identical. Track the
   companion peaks the way `_peak_courant` is tracked. **The `.copy()` is load-bearing
   (§2.3)**, and the observer's own aliasing gate transfers directly: 1.4e-12 m³.
5. **`solver/run.py`** — the warning: **trigger unchanged**, message leads with the
   in-regime peak and the over-gate cell count, still names `interval_s`, still prints
   unconditionally. Keep the "check the bed against a longer interval" remedy — step 1
   re-confirmed it. The verbose `bed courant` line gains the same breakdown.
6. **Tests.** Unit tests for the helper: one hot cell that moves no bed (so
   `courant_moving` drops it while `courant` does not), one where the bed change *is*
   in the hot cells (so nothing drops), and one below the regime floor (so
   `courant_in_regime` drops it while the other two do not). The pair-or-better is
   the argument; a single arm passes by construction. Plus a shape test that the
   record and `.zattrs` carry the new keys, and that `courant` is unmoved.
   **Not** a "demo-shaped case reports a share near zero" test — measured 0.495, so
   that expectation was backwards.
7. **The before/after run** (§4) and the docs — including the roadmap's item 3, whose
   stated reason is now known to be false, and its headline figure.

---

## 4. Validation

**The keystone is that nothing moved.** This is a pure observer, so:

- `reach_alluvial` on CUDA and CPU: `bed_change` field **bit-for-bit identical** to
  the before run, gross volume identical, water mass residual identical, sediment
  residual identical. Not "within tolerance" — *identical*. If it is not, the change
  leaked into the physics and that is the whole finding.
  The baseline store for that comparison already exists: the §3 step 1 observer run
  wrote one at `M:\claud_projects\temp\morph-courant\alluvial.zarr` under an unmodified
  solver. Reproducing the recorded *figures* (168 557 m³, 2.66e-07, 4.21e-17) is
  consistent but is **not** the byte comparison this bullet asks for.
- The full suite green with **no gate's tolerance touched**, and — since §2.4 froze
  the trigger — with **no test re-homed either**. The scheduler pass set the standard:
  run the changed code against the untouched suite *first*. Here the bar is higher
  than it was when this plan was written: **any failure at all is a real one.**
- `test_an_interval_that_moves_the_bed_a_cell_per_activation_is_caught` still asserts
  exactly what it asserted, on an unmoved `courant` value. This is the fixture that
  proves the Courant gate and the celerity gate are not substitutes — it is the one
  thing this pass is most likely to damage silently.

**The numbers table** is §1.3, produced before any code changed. What the after-run
must show is that the shipped helper reproduces those columns from inside the solver
— same four runs, same figures — and that the `courant` column did not move.

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
  different reasons. **Still carried, deliberately** — and §1.3 raised its stakes: the
  understatement is *why* the share points the wrong way on the rain sheet, so it is
  no longer just an incompleteness, it is the reason a whole class of trigger is
  unavailable.
- **`c_b` stays a rigid-lid upper bound.** §1.2's mechanism is not fixed, it is
  documented. A slenderness-aware celerity is a change to the *reference* the
  celerity gate is measured against, which is a physics change with its own
  validation, not a diagnostic pass.
- **`MORPH_COURANT_GATE` stays at 1.0.** The Courant-3.30 fixture is direct evidence
  that 3.3 is already broken *where the reference is valid*. Raising the constant to
  make the demo quiet would be calibrating a threshold by the answer it gives on one
  scenario — the opposite of what this pass is for.
- **No relative-submergence guard**, only a relative-submergence *report*. §2.1 says
  where the line is: `MORPH_REGIME_FLOOR` is printed and stored, and nothing branches
  on it. The moment anything does — a gate, a trigger, a clamp — M7 step 8's objection
  returns in full, and `_sediment_bowl`'s docstring already states the contingency:
  the tests built on it must be re-homed to channel flow, not weakened.
- **The demo stays loud, and that is now the finding rather than the defect.** After
  this pass `reach_alluvial` still prints a warning. What changes is that the warning
  is legible: it says 19.4 over 1414 cells with 578 over the gate, not 39 271. The
  remaining overstatement is §1.2's rigid-lid mechanism, which is a property of the
  reference and is carried, in writing, in place of the vaguer *"the diagnostic
  overstates"* the roadmap carried before.
