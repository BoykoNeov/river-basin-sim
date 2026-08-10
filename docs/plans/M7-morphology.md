# M7 — Morphology (Exner + transport capacity on the slow clock)

The last milestone on the roadmap. The bed stops being static: sediment is
entrained where the flow can carry it, deposited where it cannot, and the
difference moves `z`. HANDOFF §9 names the scope in five words — *"sediment
transport (Exner + transport capacity) on the slow clock"* — and §8 has been
holding the seat since M5: *"slow processes (reservoir daily rules, sediment
morphology on a long clock) advance at sync points via operator splitting."*
`solver/scheduler.py` already schedules them; `solver/processes/reservoir.py`
already shows what one looks like. M7 is the second consumer of that seam, and
the first one that writes to a field the flood scheme reads every step.

Read §0 before anything else. The scope fence is most of this plan's value: a
morphodynamics milestone can absorb an unbounded amount of work, and this one is
deliberately the equilibrium-bedload case, on one scheme, with a frozen channel
section.

---

## 0. Scope — what M7 is and is *not*

**Is:**

- A **capacity-based bedload law** (Meyer-Peter–Müller) evaluated from the flow
  the solver already has, per cell, per fast step.
- **Exner** bed update, `∂z/∂t = −1/(1−p) · ∇·q_s`, applied on the **slow clock**
  by operator splitting, from a transport flux integrated over the whole slow
  interval rather than sampled at its end (§1.3).
- A **sediment mass ledger** with its own relative gate, mirroring
  `massbalance.py` — including the causal peak floor, because an erode-then-fill
  run collapses the denominator exactly the way M4's drain-to-empty run did.
- `bed_change (T, Y, X)` in the canonical store (§7.2 addition), a `[sediment]`
  stanza in §7.1, and a `scripts/plot_bed_change.py` demo figure.

**Is not** — each of these is a real thing someone will expect, and each is
declared out, loudly, in code and in the docstring:

- **No suspended load, no washload, no non-equilibrium adaptation length.**
  Transport is at capacity, instantaneously, everywhere. That is what "Exner +
  transport capacity" means and it is a known, stated simplification: it
  over-sharpens bed features and cannot represent fines travelling through a
  reach. Non-equilibrium exchange needs a second distributed field and its own
  advection, which is a milestone, not a step.
- **No graded sediment.** One `d50`, scalar or field. No size fractions, no
  armouring, no hiding/exposure correction.
- **No bank erosion, no planform change, no channel migration.** Exner moves the
  floodplain bed `z`; see §1.5.
- **The sub-grid channel section `(w, d)` is frozen.** `z` moves, so the channel
  invert `z − d` translates *with* the bed and the section shape is fixed. This
  is a physical claim, not an oversight — §1.5 says why.
- **HLLC is not supported.** LI-only, loudly, in the voice M5 used for
  `fixed_stage` (HLLC-only) and M6 for channels (LI-only). §1.4.
- **The viewer does not animate terrain.** §1.7.

---

## 1. Design decisions

### 1.1 The bed change accumulates in float64, because float32 `z` cannot hold it

This is the tightest constraint in the milestone and it decides the state design,
which decides the store contract, which decides the viewer. It was **measured
before the plan was written**, not assumed.

`z` is an absolute elevation, O(100 m) even after `[grid] datum` moves the floor
(`resolve_datum` moves the origin, not the range). A morphological increment is
sub-millimetre per activation. The question is whether the increment survives the
add at all. Measured on `scenarios/reach_basin.toml` — the real mosaic, coarsened
to the run grid, with the real channel fields and real bed slopes — with MPM at a
900 s activation interval and a 10% divergence of `q_s` across one cell:

```
bed true      : 84.9 .. 260.0 m      datum z_ref = 84 m
bed stepping  : 1.0 .. 176.0 m       eps(z_max) = 1.526e-05 m
channel cells : 2232   w = 6.8..45.0 m   bed slope p10/p50/p90 = 1.15e-3 / 1.94e-3 / 3.52e-3

regime                          h (m)     q_s (m2/s)   dz/act (m)   dz/eps
channel gravel  Q=5   w=32m     0.261     1.004e-04    1.506e-04       9.9
channel gravel  Q=25  w=32m     0.693     6.245e-04    9.368e-04      61.4
channel gravel  Q=60  w=32m     1.186     1.512e-03    2.268e-03     148.6
channel sand    Q=25  w=32m     0.693     7.310e-04    1.097e-03      71.9
floodplain sheet h=20mm         0.020     2.655e-07    3.983e-07       0.0
floodplain sheet h=100mm        0.100     2.167e-05    3.250e-05       2.1
floodplain sheet h=20mm (silt)  0.020     1.938e-06    2.907e-06       0.2
floodplain sheet h=100mm (silt) 0.100     2.661e-05    3.992e-05       2.6
```

The discriminator: an increment must clear `eps(z)` by ~10× for float32
accumulation to be trustworthy. **The channel at low flow sits exactly on that
line (9.9) and the floodplain fine-sediment regime is an order below it (0.2).**
Below 1.0 the add is a no-op: float32 `z += dz` discards the entire increment,
every activation, forever. That is not drift you can characterise — it is a
term silently deleted from the physics, and it is a full order worse than the
conditioning that motivated the precision pass.

**Decision: bed change lives in a float64 `(ny, nx)` accumulator `dz_cum`, and
`z` is *recomputed* as `float32(z0 + dz_cum)` at each activation** — never
incremented in place. The precedent is already in the tree and it is not
`h_comp`: it is `loss_cum`, whose docstring says the same thing for the same
reason (*"sink outflow can concentrate at a single cell and grow far larger than
any per-step increment; a float32 accumulator there would drift"*). HANDOFF §2
constrains the *stepping* fields; `loss_cum` established that a float64
grid-sized accumulator sits outside that fence, and this is the same kind of
object — a ledger, not a state variable.

Recomputing rather than incrementing buys two more things: `z0` keeps its exact
identity for the whole run (so the stored `bed` and the stored `bed_change` are
consistent by construction rather than by accumulation), and the sediment ledger
gets its volume term, `Σ dz_cum · A · (1−p)`, in float64 for free.

**On the roadmap's carried requirement.** M6 left an instruction: *"M7's sediment
must go through `sources.py` rather than a bare `+=`."* The float64 accumulator
is the stronger form of that requirement, not an exemption from it — in f64 at
`|dz| ~ 1e-2 m` the ULP is 1.7e-18 m, so compensation buys nothing on top. What
M7 owes `sources.py` is the *idiom*: the exact-accumulation helpers move there
(generalised to take a target array instead of hard-coding `state.h`), so there
stays exactly one module that owns "how a distributed quantity is accumulated
into a field", and the next milestone finds it there. The `[sediment]` path gets
the ratio-between-configurations gate and the arm-is-live canary that HANDOFF §10
requires of any such claim.

### 1.2 Transport: MPM, and the law is the thing the gate is written against

`q_s = 8 (θ − θ_c)^1.5 · sqrt(s' g d50³)` per unit width, with
`θ = τ/(ρ s' g d50)`, `s' = 1.65`, `θ_c = 0.047`, and zero below threshold.
Direction is the flow direction; the flux is evaluated **on faces**, in the same
staggered layout as `qx`/`qy`, so `∇·q_s` is a face-difference and the update is
conservative by construction the way the water continuity update already is.

Shear comes from the friction the scheme already computes — `τ = ρ g h S_f`, and
`friction.manning_denominator` / `manning_denominator_radius` already own the
Manning form for the floodplain and the channel face respectively. Reusing them
is not just economy: it means a channel face's transport sees the hydraulic
radius `A/P`, not the depth, exactly as its conveyance does.

MPM is chosen over Engelund-Hansen or van Rijn for one reason beyond
familiarity: **the threshold is a cheap, sharp test.** A run below `θ_c`
must produce a bit-exact zero bed change, and that single assertion catches a
sign error, a units error, or a velocity-independent term wired in by accident.

The law is a `@wp.func` behind a name, so a second law is an addition rather than
a rewrite — but M7 ships one, and the celerity gate (§3) is derived for *that*
one. Gating against an external flume dataset that cannot be verified from here
would repeat M5's EA Test 1 cost (*"faithful in form, reconstructed in detail"*);
gating against the analytical celerity of the law actually implemented is
self-consistent, checkable in-tree, and still discriminates.

### 1.3 Transport is integrated over the interval; the bed moves at the activation

The reservoir samples its pool stage at the activation instant and multiplies by
the interval. That is correct for a control law and wrong for sediment: `q_s`
goes as roughly `h^1.5 S_f^1.5`, so sampling a passing flood wave at one instant
and scaling by 900 s misrepresents it badly in both directions depending on where
the sample lands.

**So the fast loop accumulates, and the slow clock applies.** A face-sized
accumulator `qs_int` gains `q_s(t) · dt` every fast step; at each activation the
morphology process forms `Δz = −1/(1−p) · ∇·qs_int / dx`, adds it into `dz_cum`,
rebuilds `z` and `eta`, and zeroes `qs_int`. The bed still changes only on the
slow clock — the splitting is intact — but what it changes *by* is a proper time
integral rather than a point sample.

**`qs_int` is float32 with Kahan compensation, not float64 — and that is the
opposite call from `dz_cum` for a reason.** Its increments `q_s·dt` are the same
order as the accumulator itself over one interval, which is exactly the regime
Kahan was built for and exactly what makes it *fail* on `z` (§1.1: there the
increment is orders below the accumulator's ULP, so there is nothing to
compensate — the add simply does not happen). So this is where the roadmap's
carried instruction lands literally: `qs_int` accumulates through
`solver/core/sources.py`, HANDOFF §2 keeps its float32 fields, and the one f64
grid array in the milestone is the one that is unambiguously a ledger. Cost:
four float32 face arrays (`qs_int` x/y plus their compensation terms). Extend
`run.py::field_memory_mb` to count them, since its whole purpose is printing the
working set before a reach-scale run discovers it as a CUDA out-of-memory.

The property this buys is testable and is the milestone's cleanest gate: **halve
the activation interval and the answer must barely move** (M5's "activations are
bit-identical at `dt_max` 5.0 and 1.0" is the same shape of claim). Under
instantaneous sampling it would not.

Cost: one extra full-grid kernel per fast step, armed only when sediment is
configured. On 768² that is noise beside the LI flux kernels, and every run
without `[sediment]` keeps the M6 kernels and stays **bitwise unchanged**.

**The morphological CFL is a real limit and gets a loud pre-run print.** A bed
wave that travels more than a cell per slow interval is a splitting artefact, not
a result — the exact analogue of the reservoir's *"54,000 m³ into one 40 m cell is
a 34 m column"*. Print `c_bed · interval_s / dx` before stepping, from the
scenario's own numbers, in the codebase's existing habit of saying the dangerous
ratio out loud before it bites.

### 1.4 LI-only, loudly

The precedent is established twice: `fixed_stage` is HLLC-only and says so with a
`ConfigError`; sub-grid channels are LI-only and say so. M7 takes the same shape,
and the reason is specific: **sub-grid channels are LI-only, and morphology in a
river happens in the channel.** An HLLC-only M7 could not touch `reach_basin` at
all, which would strand the milestone from the demo that makes it worth having.

`scheme = "hllc_fv"` + `[sediment]` is a `ConfigError` naming both, not a silent
degradation. HLLC morphology is a later milestone if it is ever wanted; the
hydrostatic-reconstruction well-balancedness argument would have to be re-made
against a bed that moves, which is not a small re-derivation.

### 1.5 Exner moves `z`; the channel section is frozen; structures do not erode

Three separate fences, each of which someone will otherwise read the wrong way.

**The channel.** M6's channel is `(w, d)` and a storage curve `h → η` derived from
them. If `d` evolved, the storage curve would change shape mid-run and the CFL
derivation would re-open (*"the CFL reduces over the water column, since a channel
concentrates depth by dx/w"*). So: Exner moves `z`, the invert `z − d` translates
with it, and the section shape is fixed. A channel that aggrades rises bodily;
it does not fill in. That is a genuine limitation of the parameterisation and it
goes in the docstring next to M6's existing honesty note.

**Structures.** `apply_barriers` raises the bed to `crest_m` before stepping.
Nothing in Exner stops the flow from scouring a dam out from under its own
`target_stage` release rule, which would be a spectacular and entirely spurious
result. **Structure cells are masked out of the bed update** — a dam is
engineered, not alluvial.

*Masked at the cell, not at the face*, and the distinction matters. The
divergence `∇·qs_int` is formed from real face fluxes everywhere, including at a
structure cell; the mask then discards that cell's `Δz` and banks exactly the
discarded amount, `Δz · A · (1−p)`, into the sediment ledger. That is trivially
exact and it is what the sentence above means. Zeroing the faces bounding a
structure instead would make the dam a no-flux sediment wall — a different and
arguably better model, since it also stops sediment routing *through* the
structure — but it changes the neighbouring cells' divergence too, so it is a
different physical statement and not the one M7 makes.

**Any other limit is banked too.** An alluvium floor ("do not erode below
bedrock"), a no-negative-deposition rule, anything that stops the bed moving as
the law says — the amount not applied is banked into the sediment ledger, never
silently clamped. This is the direct analogue of the rule that already governs
depth: *"never keep depth non-negative with a bare `max(h, 0)` — it invents mass"*.
The same sentence with `z` in it is just as true.

### 1.6 Deposition can dry a cell, and the rule for that is chosen, not discovered

`z` rises at fixed `h`, so `η = z + h` rises with it and the *water* is unchanged
— but a cell can go from wet to effectively dry as its bed comes up through a thin
sheet, and the storage curve's `h_bf` boundary can be crossed in a channel cell.
This is the wet/dry equivalent of the shoreline case, and the shoreline case is
the one that has already caught a real bug once (M4: MUSCL across a dry neighbour
spun a bowl to ~20 m/s).

**The rule is: there is no special rule.** `h` is untouched by a bed update, `η`
rises with `z`, and the existing `H_DRY` guard already decides what "dry" means
everywhere else in the solver. Nothing is added — but it is *asserted*, with a
test that deposits a cell's bed up through a thin sheet and checks that the
volume is still there and the guard behaves, and by re-running
`test_shoreline_lake_at_rest_on_bumpy_bed` after every change in this area. The
point of naming a no-op rule is that the alternative — a special case someone
adds later to make a picture look right — is exactly how `max(h, 0)` got into
M4.

Water volume is conserved through a bed update by construction — `h` is untouched
and `V = h·dx²` — which is the same argument that made M6's channels exactly
mass-conserving. The water ledger must stay green *and* be checked, because "the
bed update quietly ate water" is precisely the failure this milestone can produce.

### 1.7 The store carries `bed_change`; the viewer does not animate terrain

§7.2 already reserved the room: *"`bed` (Y, X) float32 bed elevation z (static
unless morpho)"* and *"[sediment, ...] added in later milestones"*.

**Store:** `bed` stays static and stays the **initial** bed, un-shifted, exactly as
today; morphology ships as `bed_change (T, Y, X) float32` beside it. Two reasons
to add rather than promote `bed` to `(T, Y, X)`: every existing reader — the
viewer, `--rbverify`, the tests — keeps working untouched; and `bed_change` is a
*small* quantity, so float32 storage of it has full relative precision where
float32 storage of `z` would not. `unshift_bed` is irrelevant to a difference,
which removes the mixed-datum trap that promoting `bed` would have created —
though the initial `bed` must keep going through it as it does now.

**Viewer:** unchanged in M7, and honest about it. The terrain shown is the
initial bed; a store carrying `bed_change` gets a note in the manifest and the
viewer says so rather than implying the terrain is current. Animated terrain
means re-fitting `bed_tex` and the height map per frame in `_apply_geometry`,
which is a viewer milestone, not a rider on this one — and the **existing carried
debt makes it worse, not better**: the shader still lifts water as `bed + depth`
rather than through the storage curve, so a moving bed would move that mis-lift
too. Fix the lift first, then animate.

The demo evidence is therefore a figure, not a scrub: `scripts/plot_bed_change.py`
renders initial bed, final bed change, and the sediment-ledger series to a PNG.

---

## 2. Build order

Each step keeps `ruff` + `pytest` green and is one commit. Steps 1–3 land the
contract and the arithmetic with no physics; step 4 is where water first moves
sediment.

1. **`[sediment]` in `solver/io/config.py`** — `d50`, `porosity`, `interval_s`,
   `law`, optional `alluvium_thickness`, scalar-or-field per the existing
   `_scalar_or_field` idiom. `ConfigError` for `scheme = "hllc_fv"` + sediment.
   Coarsening rules declared for every new field in `coarsen.py` (`d50` by mean;
   an alluvium thickness by mean — a *thickness* is volume-preserving, unlike
   M6's channel width). Config tests only.
2. **`solver/core/sediment.py`** — the MPM `@wp.func`, the face-flux kernel, the
   Exner divergence kernel, the `z`-rebuild kernel. Pure kernels, unit-tested
   against hand-computed values on tiny grids, no run-loop wiring. Includes the
   **analytical celerity helper `c_b = (1/(1−p))·dq_s/dz`, built once and used
   twice**: it is what §3's bed-wave gate compares against *and* what §1.3's
   pre-run morphological-CFL print is computed from. Neither is free without it,
   which is why it lands this early.
3. **The celerity test scenario** — a small synthetic straight channel, sized so a
   bed bump migrates several cells inside the run. Built here, not at step 7,
   because sizing it is a derivation (`c_b`, run length, cells per wave) and
   discovering it late means redoing the geometry. It is **not** the demo, and it
   is not `reach_basin`: see the note in §4 on why the demo cannot carry this gate.
   **Done** — `validation/bedwave.py` (the durable fixture: geometry, the derived
   design point, three migration estimators) + `validation/test_bed_wave.py` (a
   *provisional* harness, superseded by step 5's process, and four checks). No
   `.toml`: every gate in this harness builds its state in Python, and `run.py`
   refuses `[sediment]` until step 5. What sizing turned up, all measured:
   - **The design point** — 240×1 cells @ 2.5 m (600 m), `S = 0.002`, `n = 0.035`,
     `d50 = 8 mm`, `q = 2.5 m²/s` → `h_n = 1.4959 m`, `Fr = 0.44`,
     `θ = 0.2266` (4.8 θ_c), `c_b = 8.63e-3 m/s`. A 15 mm × σ 15 m bump migrates
     **16 cells (40 m) in 4635 s**, 103 activations at 45 s (Courant 0.155),
     ~13 000 fast steps, ~2 s on CPU. The solver hits that design point to four
     decimals (1.4959 m, `q = 2.5000`, θ = 0.2266), so the gate's denominator is
     the celerity of the flow that actually ran, not a nominal one.
   - **Measured migration: 0.993 c_b** by cross-correlation lag (gated), 1.004 by
     crest fit, 0.942 by centroid; crest keeps 0.72 of its height; signal/background
     ≈ 30. Interval stability (xcorr/`c_b`): 0.93 @ 11.25 s, 0.95 @ 22.5 s,
     **0.99 @ 45 s**, 0.99 @ 90 s, 0.97 @ 180 s, 0.94 @ 360 s (Courant 1.24) — a 7%
     spread, so **step 8 can gate ±20% on the shape estimator** and treat the crest
     and centroid as printed diagnostics.
   - **The reference is only meaningful for a slender bump.** `bed_celerity`
     linearises at a *rigid* water surface; how close reality gets is set by
     `σ/(h/S)`. At 0.02 the wave lands within 1% of `c_b`; at 0.07 it reads ~0.9
     with a visible grid dependence; at 0.30 it collapses to 0.37 and refining `dx`
     does not recover it. Two design points were rejected on exactly this
     (`Fr = 0.77` at `σ/(h/S) = 0.065`; a shallow steep flume at 0.30).
   - **The ends must be pinned, and that is a sediment BC.** Boundary faces carry no
     bedload, so with free ends the inlet scoured **1.12 m** and the outlet grew a
     **0.91 m** sill in 26 activations, whose backwater lifted the reach 1.495 →
     1.944 m (30%) — the flow is wrong everywhere before the bump has moved. Pinning
     one cell at each end via `exner_update`'s `dz_lo == dz_hi == 0` makes them an
     equilibrium feed and an equilibrium sink (banked, not clamped), and the reach
     holds 1.4952 m. See the step-5 requirement below and §4.
4. **`state.py` + `sources.py`** — arm `dz_cum` (**f64**, cell-centred), `qs_int`
   + its compensation (**f32**, faces, §1.3), `z0` (the pristine initial bed);
   generalise the exact-accumulation helpers to take a target array; extend
   `field_memory_mb`. Assert the bitwise-unchanged invariant for every existing
   scenario with sediment unarmed.
   **Done** — `sediment.arm_sediment` / `SedimentState` (attached to
   `State.sediment`, the `arm_hllc`/`arm_channels` idiom), and the arming is read
   off step 2's kernel signatures rather than off this prose: **seven** arrays, not
   §1.3's four, because §1.3 counts only the face set. Four decisions worth naming:
   - **The split between state and process is by clock.** `d50` is read every *fast*
     step, so it is state, exactly as M6's channel geometry is; `interval_s`, the
     alluvium thickness and the `dz_lo`/`dz_hi` bounds are read only at an
     *activation*, so they belong to the step-5 process. That is not tidiness —
     putting config-derived bounds on `State` would build the very trap step 5's
     first requirement warns about, since nothing in `[sediment]` can hold an outlet
     cell *down*. The celerity fixture keeps owning its own pinned ends and proves
     the seam works before step 5 exists.
   - **`field_memory_mb` sums the two widths separately**, not "count × 4 bytes":
     six of the seven are f32, but `dz_cum` + `dz_unapplied` are f64, and at 768²
     that **pair of two** weighs exactly what the **four** face accumulators do
     (~9 MB each) — a 4-byte hard-code would have under-reported morphology by a
     third, hiding precisely the arrays §1.1 forced to f64.
   - **Arming is deliberately *not* idempotent.** Re-arming would recapture `z0`
     from a bed that has already moved and zero the change that moved it — a silent
     ledger reset leaving `bed_change` a lie. It raises instead. (`z0`-after-barriers
     stays a run-loop obligation the kernels cannot check; step 4 pins it in the
     docstring and in a test that moves the bed after arming and rebuilds.)
   - **The "generalise the exact-accumulation helpers to take a target array" item
     was dropped, not deferred.** `kahan_add` is already a generic `@wp.func`, the
     rain kernels already take `h`/`comp` as parameters, and `sediment.py` already
     imports it — §1.1's actual requirement (one module owns *how* a distributed
     quantity is accumulated into a field) holds today. Sediment's increment is
     computed per-face from the flow and fused into its own kernel, so it cannot use
     a generic wrapper: the helper would have shipped with zero callers.

   - **A zero `d50` does *not* make a cell immobile, and the first draft's label said
     it did.** `d50` is face-averaged like `n`, so a lone zero cell's faces carry half
     its neighbours' grain size and move its bed like any other — measured, −0.077 m
     in one activation — and the error has a sign, since `θ ∝ 1/d50` makes an isolated
     zero read as *more* mobile. Only a contiguous zero region's interior faces carry
     nothing. The count is kept (it is useful) but named `zero_d50_cells` for what it
     is, and a test pins the real behaviour. Immobilising a cell is
     `alluvium_thickness = 0`, whose bound also *banks* what it refused. Same class as
     `e8c6d85`: a docstring that inverts a meaning right before step 5 reads it.
   - **`arm_sediment` now enforces the same open `(0, 1)` porosity interval `[sediment]`
     does**, so a directly-armed fixture cannot be legal where a scenario is refused.

   **Verified by repointing the step-3 fixture** at the arming path rather than by a
   fresh unit test — `validation/test_bed_wave.py` hand-rolled exactly these seven
   arrays, and every recorded number reproduces bit for bit (xcorr **0.993 c_b**,
   crest 1.004, centroid 0.942, crest retention 0.72, signal/background 29.7,
   free-end inlet −1.1226 m / outlet +0.9071 m / reach 1.9439 m, pinned ends exactly
   0.0). At this step the bitwise invariant is *structural* — unarmed means the
   attribute is `None` and nothing is allocated — with the suite's stored baselines
   as the regression. **292 tests green.**
5. **`solver/processes/morphology.py`** — the slow process, modelled on
   `reservoir.py`: constructed after `State`, exposes `as_slow_process()`,
   `advance(t, dt_slow)` returns a record, `series` lands in `.zattrs`. Plus the
   per-fast-step accumulation hook. Two requirements step 3 hands it:
   - **It must accept explicit `dz_lo`/`dz_hi` arrays (or a frozen-cell mask) at
     construction**, not only bounds derived from `[sediment]`. `alluvium_thickness
     = 0` pins the floor and leaves the ceiling open, so nothing in the config can
     hold an outlet cell *down* — and step 3's fixture needs exactly that, as will
     any scenario that wants an equilibrium sediment BC.
   - **Assert once that the in-`step` hook is equivalent to accumulating after
     `step` returns**, which is what the step-3 harness does and what makes its
     recorded numbers transferable: for LI, `eta` is computed at the top of the step
     and never rewritten, the faces are final after the limiter, and the
     post-continuity sinks touch only `h`. A future reordering would silently move
     the gate.

   **Done** — `MorphologyProcess` + `bed_change_bounds`, `sediment.accumulate_transport`,
   and the run-loop wiring; the step-1 `NotImplementedError` is inverted into an
   end-to-end run. Six decisions worth naming:
   - **The hook is *not* in `run.py`, and this bullet's own text was the weaker
     statement.** "Beside the inflow injector" points at code that runs *before*
     `scheme.step` — the wrong anchor entirely for reading post-limiter faces. Step 2's
     committed `sediment.py` docstring is the later and more specific instruction and
     it names the place: after `limit_qx`/`limit_qy`, before `update_h`. So the
     accumulation is driven **off `state.sediment` inside `local_inertial.step`**, the
     sixth optional branch on a path that already reads `channels`/`h_comp`/`rain`/
     `infil`/`open_edges` the same way. No signature change, nothing LI-only leaking
     through the `scheme.step` dispatch seam, and the bitwise invariant stays
     *structural* — unarmed means `state.sediment is None` and no kernel is launched.
     `run.py` still owns constructing the process and driving the activation half.
   - **The equivalence is asserted twice, and the second one is the evidence.**
     Directly, in `test_morphology`, in-step vs the step-3 pattern (accumulators
     allocated, then hidden from `step`) over 12 steps with **channels + rain +
     infiltration + an open boundary** all armed — precisely the tail kernels that
     could falsify it — bit-identical on both arms, plus the depth field bitwise equal
     to a run that never armed morphology. And empirically, by repointing the step-3
     fixture at the real process: **every recorded number reproduces bit for bit**
     (xcorr 0.993 c_b, crest 1.004, centroid 0.942, retention 0.72, signal/background
     29.7, free ends −1.1226 / +0.9071 m and a 1.9439 m reach, pinned ends exactly 0).
     The write-sets were *read* before the test was written, not assumed:
     `update_h`, `sources.*`, `apply_infiltration` and `apply_open_outflow` touch only
     `h`, `h_comp` and `loss_cum`.
   - **The fixture's warm-up flag disappeared rather than moving.** State-armed
     accumulation means "start transporting now" is spelled `arm_sediment` *at* the
     warm-up boundary, where `z0` is still pristine because nothing has moved the bed.
     One less enable to keep in sync with a `transporting` predicate.
   - **The morphological Courant number is measured, not assumed** —
     `sediment.celerity_field` evaluates `bed_celerity` per cell from the flow the
     state actually has (channel cells through the channel form: its own discharge,
     column depth and `w/dx` fraction, because reading them as floodplain would
     under-report by orders of magnitude exactly where the transport is), and every
     `BedChangeRecord` carries the peak. There is deliberately **no pre-run print**:
     at `t = 0` a scenario usually has no flow, and a Courant number from a flow that
     has not happened is a reassurance, not a warning. Gating it is step 8. First
     cross-check: the fixture measures **0.163 against its independently derived
     0.155**, the 5% being the bump crest, where the bed really is locally faster.
   - **Two obligations that were easy to drop, and one silent-failure trap.**
     Structure cells are frozen (`dz_lo == dz_hi == 0`) from `elev.structures`, or the
     flow scours a dam out from under its own release rule — §1.5, and not restated in
     this bullet. And the alluvium thickness must be read **field before scalar**: a
     field-backed floor leaves `alluvium_thickness_m` at the unused `0.0` fallback that
     `load_field` needs, which read on its own says "bedrock everywhere" and freezes
     the whole bed with no error anywhere.
   - **`eta` is refreshed with the bed, and reservoirs advance first.** The next fast
     step recomputes `eta` anyway, so the extra launch changes no physics — but a state
     whose `eta` disagrees with its own `z` is a trap for anything reading between
     ticks. On a tick where both processes are due, the release rule runs first: it
     reads a stage `z + h` and should read it off the bed the interval's water actually
     flowed over.

   - **Two hazards found in review, and the first one's *mechanism* did not survive
     checking.** The worry was that a coarse `dt_max` against a fine `interval_s`
     would let several intervals collapse into one `advance` — the bed jumping by all
     of them at once, exact in mass and silently coarser in splitting, which is
     precisely what step 8's interval-independence gate would then measure against
     itself. Checked rather than assumed: with `dt_fn` returning 30 s against a 10 s
     interval the scheduler still delivers **ten activations of exactly 10 s each**,
     because activations *are* sync points and M5 clamps every step to land on one.
     So the scheduler cannot produce it — but a **hand-driven** caller can, and that
     is how the celerity fixture and every future harness drive this process, so
     `advance` now refuses a `dt_slow` longer than its configured interval. It caught
     two of this step's own tests immediately.
   - **The celerity diagnostic takes the larger of its two components, not the
     channel one.** A channel cell flowing overbank carries a real floodplain flux
     across most of its width; selecting the channel component would report exactly
     zero for a cell whose channel happened to be still — the wrong direction for a
     warning to fail in, and `reach_basin` scale is exactly the overbank case.

   **305 tests green.**
6. **`SedimentLedger`** in `massbalance.py` or beside it — same shape as
   `MassLedger`, same Kahan f64 accumulators, same causal peak floor, its own
   relative gate. Masked-cell and clamped transport bank here.
   **Done** — `SEDIMENT_GATE` + `SedimentRecord` + `SedimentLedger` in
   `massbalance.py` (so one module owns both credibility gauges), recorded at the
   output cadence from `run.py` and shipped in `.zattrs`. Five decisions worth naming,
   three of which are places this step's own brief was wrong:
   - **"Same Kahan f64 accumulators" does not survive contact with the terms.** Every
     sediment quantity is a *fresh full-field f64 reduction* at each accounting point
     — `Σ dz_cum`, `Σ dz_unapplied`, `Σ|dz_cum|` — not a running sum of many small
     increments, so there is nothing for `_Kahan` to compensate and it is deliberately
     unused. What the water ledger lends is the **idiom**: a record dataclass, a causal
     peak, a relative gate, `as_attrs`. Shipping the accumulator anyway would repeat
     exactly what step 4 refused when it dropped the generalised accumulation helper.
   - **There are no inflow/outflow terms, and that is the physics, not an omission.**
     Bedload cannot cross a domain edge — `accumulate_qs_*` launch over interior faces
     only, so the four edge face-rows stay exactly zero and `div(qs_int)` telescopes to
     nothing. The whole balance is therefore **`Σ dz_cum + Σ dz_unapplied = 0`**: every
     metre gained somewhere came from somewhere else, plus what the bounds refused. A
     bound is a *supply* (`supplied = -banked`): the fixture's frozen inlet that wanted
     to erode 1.12 m and did not has fed the domain that much solid. Accumulators for a
     boundary term nobody can produce would ship with zero callers;
     `test_the_boundary_faces_carry_no_bedload` is the standing evidence instead, and
     it is what fails first if a future supply BC starts writing them.
   - **The causal peak is on the *gross*, and it is the primary scale rather than a
     floor.** §3 justified the floor by "erode-then-redeposit nets to zero" — but for
     sediment the net is *identically* zero, since that is the invariant under test, so
     a peak-of-net would be vacuous and the denominator would end up being the residual
     itself. The scale that means anything is `A(1-p)·Σ|dz_cum|`, and it is **reported
     in every record** for the same reason: without it a near-zero residual cannot be
     told from "nothing happened" (`test_the_gross_volume_tells_a_balanced_bed_from_a
     _still_one` pins both readings side by side).
   - **What is balanced is `dz_cum`, not `z`.** The float32 bed is a *rendering* of the
     f64 change (`z = float32(z0 + dz_cum)`); differencing two O(100 m) float32
     elevations to recover a sub-millimetre change is the cancellation §1.1 built
     `dz_cum` to avoid.
   - **The gate is measured on both paths, because banking is the one that can silently
     break.** On the celerity fixture over 103 activations: worst relative residual
     **5.7e-15 with the bounds firing**, **2.1e-16 with them absent** — and the
     *absolute* residuals are 1.4e-15 and 2.5e-15 m³, i.e. the same round-off, so the
     27× ratio is only the pinned run's 27× smaller gross (1.24 vs 33.5 m³) and not a
     banking defect. `SEDIMENT_GATE = 1e-11` leaves ~3 orders over that, and is 5
     orders tighter than the water gate because this balance is f64 arithmetic over an
     f64 field where the water gate absorbs float32 flux divergence.

   `run_simulation` still returns `MassLedger` — the sediment balance travels in
   `.zattrs`, which also makes the shipped artifact the thing under test. **312 tests
   green.**

   **One finding for step 8, measured here and out of scope for this step.** The bowl
   scenario in `test_run.py` scours **5.6 cm in its first 150 s activation** and then
   almost nothing for the rest of the run — because MPM at the dry threshold is
   enormous. `τ/ρ = g n² q²/h^(7/3)` diverges as `h → H_DRY = 1e-3 m` at fixed `q`: a
   1 mm sheet at 0.5 m/s with `n = 0.03`, `d50 = 2 mm` reads `θ = 0.68`, **14× θ_c**,
   for `q_s = 1.4e-3 m²/s` — which over a 150 s interval and a 20 m cell is exactly the
   ~1.8 cm/activation observed. MPM is a channel bedload law and an overland sheet at
   the wet/dry guard is outside it; the ledger is exact through all of this (2.8e-17)
   because conservation is not the thing at stake. Step 8 has to decide whether the
   gate scenarios stay clear of that regime or the law carries a depth guard, and §4
   should carry it as a risk beside the `dx/w` one.
7. **Store + export** — `bed_change` through `ZarrWriter`, §7.2 and §7.1 edits in
   HANDOFF, the manifest note, `scripts/plot_bed_change.py`.
   **Done** — `bed_change (T, Y, X)` in `zarr_writer.py` (created only when a run has
   morphology, appended at every output frame from `run.py`), `manifest["morphology"]`
   + a viewer `_report_morphology()` print, the §7.1/§7.2/§7.3 edits, and the figure
   script. Five decisions worth naming:
   - **The keystone is `bed + bed_change[i]`, and it is consistent by construction.**
     `z0` is captured from the same array `bed` is written from (after barriers, before
     any water moves) and `z = float32(z0 + dz_cum)` is rebuilt from it at every
     activation — so the two stored arrays add up because of §1.1's design, not because
     the writer arranges it. `bed_change` deliberately does **not** go through
     `unshift_bed`: a *difference* of elevations has no origin to move, which is what
     lets a reader add the pair without knowing the datum. The discriminator is scale,
     not tolerance — un-shifting by mistake would offset a centimetre-scale field by
     `z_ref` (9 m in the test), three orders away from anything two datums can
     legitimately differ by.
   - **`ZarrWriter.append` refuses a mismatch in both directions**, because zero is a
     legal bed change. A preallocated frame reads as *"the bed did not move here"* —
     a statement about the physics, not a visible hole — so an omitted frame is the
     same class of quiet lie as `max(h, 0)`. Symmetrically, a store without morphology
     refuses one, so the unarmed store stays exactly M6's: no array, no attribute, and
     a byte-identical manifest (the `morphology` key is added only when the store
     actually carries `bed_change`, never as `false`).
   - **The volume cross-check had to be against the *gross*, for the reason step 6
     found.** `Σ dz_cum` is identically zero in a domain closed to bedload, so the
     first draft of the keystone test compared two near-zeros (−6.7e-7 against
     4.2e-16) and would have passed on nothing. It now compares `Σ|dz|·A(1−p)` against
     the ledger's `gross_volume`, which is the number that says the bed moved at all.
   - **`bed_change` is not exported as frame tiles, and the manifest says why.**
     `_write_field` would make it nearly free, which is exactly the temptation §1.7
     fences: the shader still lifts water as `bed + depth` rather than through the
     storage curve, so animating the bed would animate that mis-lift with it. Instead
     `manifest["morphology"]` carries the final change's extremes and the viewer prints
     them — quantitative on purpose, since *"the terrain is slightly stale"* and *"it
     scoured 20 cm"* are different pictures and only the number separates them. Same
     idiom and same reason as `_report_domain`'s gap-fill declaration.
   - **The figure reads its numbers from `.zattrs`, never from the field it draws.**
     What the ledger balances is the f64 `dz_cum`; the stored array is its float32
     rendering, so re-deriving a volume from the picture would report the rendering's
     error as the run's. `plot_bed_change.py` also reads `attrs["n_frames"]` rather
     than the time-axis length — `finalize` records the count but never resizes, and a
     trailing preallocated frame of zeros is indistinguishable from a still bed.
     Verified by running it (`scripts/` has no in-tree tests by convention); the
     logic that is not drawing is covered by the store and export tests.

   Checked on the read path as well as in `pytest`: `--rbverify` is green on the
   existing (morphology-free) demo export — the note stays silent — and on a
   morphological export it prints *"terrain is the bed at t=0 -- this run moved it by
   −0.206..+0.090 m by t = 900 s"*. **321 tests green.**
8. **Validation** — §3, every gate.
   **Done** — `validation/test_morphology_gates.py` (the threshold pair, the
   deposition rule, the regime check), the celerity gate promoted in
   `validation/test_bed_wave.py` with interval independence and the morphological-CFL
   assertion beside it, `drive`/`Run` promoted into `validation/bedwave.py`, and
   `MORPH_COURANT_GATE` + an unconditional `run.py` warning. Six decisions worth
   naming, two of which are places this step's own brief was wrong:
   - **"Halve `interval_s`; the final bed must agree" does not hold, and the reason is
     not morphology.** Measured, shortening the interval makes the bed *worse* — on a
     flat reach with the bump removed, where every millimetre is spurious: ±0.16 mm at
     90 s, ±0.11 mm at 45 s, **±8.85 mm at 22.5 s**, **±29.3 mm at 11.25 s**, against
     a 15 mm bump. The cause is the **sync-point `dt` clamp**, and it is a
     *water-solver* artefact: see §4, *"a clamped step is not a free step"*. So the
     gate is asserted over the band the Courant number admits — 45 s against 90 s
     applied to the **same** 4680 s of morphology (104 activations against 52), which
     agree to **0.300 mm on a 10.76 mm signal, 2.8%**, gated at 5%. Matching the span
     was worth doing and worth measuring: it accounts for 0.02 mm of the 0.32 mm the
     mismatched comparison reads, not the ~0.24 mm the arithmetic predicted.
   - **The celerity tolerance was promoted, not tightened.** ±20% on the shape
     estimator is what the sizing evidence supports (a 7% spread across a 32× range of
     intervals); shrinking it to ±5% because the design point reads 0.993 would fit
     the gate to one run and flake on a reduction-order change. The crest fit and the
     centroid stay printed.
   - **The Courant gate and the celerity gate are not substitutes, and that is
     measured.** At a 900 s interval the fixture runs at Courant **3.30** and the
     result is meaningless — the bump *grows* to 1.63 of its initial height, where a
     resolved bed wave can only diffuse (0.72) — yet mass reads 1.2e-8, the sediment
     balance holds, and the **celerity estimator reads 0.95 c_b and sails through
     ±20%**. Nothing but the Courant number catches it, which is why `run.py` prints
     that warning unconditionally rather than under `verbose`.
   - **The regime decision: keep the scenarios clear, do not guard the law** (the debt
     step 6 handed over). The discriminator is relative submergence `h/d50` rather
     than an absolute depth, because it is the group the law is about: the celerity
     fixture is at **187** and the threshold pair at 35–47, while the bowl's
     millimetric sheet is at **0.5** — shallower than one grain. A guard was refused
     because it would have changed what three existing tests test *without failing
     any of them*: `sed_boulders` asserts a 1 m grain size moves no bed and passes
     because `θ < θ_c`, and an `h ≥ k·d50` cut-off would have taken that assertion
     over silently. The range is enforced where scenarios are chosen instead, and
     asserted there.
   - **The threshold pair is one variable, and that had to be checked.** `θ ∝ 1/d50`
     exactly (the shear carries no grain size) and `d50` enters no hydraulic kernel,
     so `BedWave.at_shields` re-grains the *same* reach: 42.86 mm for 0.9 θ_c, 32.15 mm
     for 1.2 θ_c. Verified rather than argued — all three arms produce **bit-identical**
     depth and face-discharge fields. Under threshold: **0 cells moved**, `dz_cum`
     bit-exact zero, with the achieved θ peaking at 0.921 θ_c at the bump crest, which
     is locally shallower (the bump is deliberately kept in both arms; flattening one
     would stop the pair being the same geometry). Over threshold: 238 of 240 cells,
     the two pinned ends excepted, with the whole reach above 1.095 θ_c.
   - **Deposition cannot dry a cell directly, and the test that says it does is
     testing the wrong thing.** `h` is volume per unit plan area and no Exner kernel
     reads it, so `η = z + h` rises *with* the bed and a cell can never be buried —
     asserted immediately after the activation, where the water is bit-for-bit
     unchanged. Drying is **hydraulic and later**: the raised cell is a mound, and the
     ordinary momentum update runs the water off it (58 fast steps here) until the
     existing `H_DRY` guard applies. The content of the gate is that the water went
     *somewhere* — mass 8.7e-8, the two scoured neighbours deeper than they started —
     and a partner test pins that the drying is local, without which a solver that
     dried the whole domain would pass on mass conservation alone. The deposition is
     hand-loaded into the transport integral rather than grown from a flow, because
     depositing 5 cm onto a 2 cm sheet means running MPM in exactly the regime the
     bullet above fences off.

   Also carried, and pinned by a test rather than remembered: **the Courant
   diagnostic samples, while the transport integrates.** `celerity_field` reads the
   flow at the activation *instant*, so an interval in which a flood arrived, moved
   577 m³ of bed and drained away reports Courant **0.000** — while the same scenario
   with its forcing held on reports 5.05. The warning is a floor on what can be
   trusted, not a certificate. Making it a true interval maximum needs a full-field
   host reduction every fast step, which is not what a diagnostic should cost.

   **333 tests green.**
9. **Demo** — `scenarios/reach_alluvial.toml`, GPU sign-off, HANDOFF + CLAUDE.md
   + roadmap updates.

---

## 3. Validation plan (the credibility gates)

- **Sediment mass conservation** — the new primary gate, mirroring the water
  gate's 1e-6: `(1−p)·Σ Δz·A + banked = net sediment in − out`, relative, with a
  causal peak floor. An erode-then-redeposit run nets to zero and would otherwise
  trip on denominator collapse — the same trap M4 hit and fixed for water.
  **Landed at build step 6**, and sharper than this bullet: the right-hand side is
  *structurally* zero (bedload cannot cross a boundary face), the net is zero for the
  same reason, so the causal peak had to move onto the **gross** displaced volume and
  became the primary scale rather than a floor. `SEDIMENT_GATE = 1e-11`, measured on
  both the banking and the bounds-free path.
- **Water mass gate stays green** — on every existing scenario with sediment
  unarmed (**bitwise identical**, the invariant M4/M5/M6 all held), and on the new
  demo with it armed (within gate). This is the regression that catches a bed
  update eating water.
- **No motion ⇒ no bed change, bit-exact.** A lake at rest has zero shear, so
  `θ < θ_c` and `dz_cum` must be exactly zero. Cheap, and it is what proves no
  velocity-independent term got wired in.
- **No transport below `θ_c`.** A steady channel just under threshold moves no
  bed; nudged just over, it does. Catches sign and units errors.
- **`test_shoreline_lake_at_rest_on_bumpy_bed` stays green**, plus a
  deposition-dries-a-cell test for §1.6.
- **Bed-wave celerity against the implemented law.** A low bump in a steady
  uniform channel migrates at `c_b = (1/(1−p)) · dq_s/dz`; gate the numerical
  celerity against the analytical one derived for MPM. This is M6's fine-vs-coarse
  pattern — self-consistent, in-tree, and genuinely discriminating. The fixture and
  its sizing evidence landed at build step 3 (`validation/bedwave.py`): measured
  **0.993 c_b** by shape correlation, so the step-8 gate is ±20% on that estimator,
  with the crest fit and the centroid printed rather than gated. Note the reference
  is a *rigid-surface* linearisation — it is only the right reference while the bump
  stays slender against `h/S` (step 2 of the build order, and §4).
- **Interval independence.** Halve `interval_s`; the final bed must agree to a
  stated tolerance. This is what §1.3's time-integrated flux buys and the reason
  to build it that way. Measured on the step-3 fixture across a 32× range of
  intervals: 7% spread in the migration, degrading only where the morphological
  Courant number passes 1.
- **Morphological CFL print** is asserted, not just printed: a scenario that
  exceeds it fails a test rather than producing a plausible-looking wrong answer.

---

## 4. Risks / watch-items

- **The equilibrium assumption is the physics risk, not the code risk.**
  Capacity-everywhere over-sharpens; a knickpoint will migrate faster and stay
  crisper than a real one. Say so wherever a result is quoted.
- **A reach-scale demo shows a channel-only signal of order centimetres, and it
  cannot carry the celerity gate.** Read the §1.1 table as a *rate* table:
  `reach_basin`'s floodplain is `q_s = 0` for gravel and 4e-5 m/activation for
  silt at 100 mm depth — ~2 mm over a 12 h run; the channel at Q=60 gives
  2.3 mm/activation, ~10 cm over the run if the peak held. That is a real and
  defensible result to show, but a **bed wave needs to migrate a measurable
  number of cells**, and at these rates it will not move one. Hence the separate,
  purpose-sized celerity scenario at build step 3 — the gate and the demo are
  answering different questions and cannot share a scenario.
- **MPM at the wet/dry guard is enormous, and a rain-on-grid scenario lives there.**
  `τ/ρ = g n² q²/h^(7/3)` diverges as `h → H_DRY = 1e-3 m` at fixed `q`, so a 1 mm
  overland sheet at 0.5 m/s reads `θ = 0.68` (**14× θ_c**) for `d50 = 2 mm` and
  `n = 0.03`. Measured at build step 6: the 16×16 bowl in `test_run.py` scours **5.6 cm
  in its first 150 s activation** — the whole run's bed change — while `h_max` is still
  2 cm. This is the law being extrapolated outside its range (MPM is a *channel bedload*
  law; a millimetric sheet is not bedload), not a conservation failure — the ledger
  reads 2.8e-17 through all of it. Either keep the gate and demo scenarios in channel
  flow, or give the law a depth guard and say so; **step 8 decides, and quoting a
  floodplain bed change before it does would be quoting an artefact.**
- **The channel storage curve amplifies depth by `dx/w`**, so a channel cell's
  shear is much larger than a naive `h·S` on the cell mean. Getting this wrong
  in either direction is a factor-of-`dx/w` error — up to ~15× on this demo.
  It is the single most likely physics bug in the milestone.
- **An open boundary grows a sill, and it is big.** Boundary faces carry no bedload
  (they are never updated — that *is* the closed BC) and the local-inertial open
  boundary is a post-interior sink on the edge *cell*, not a face. So water leaves
  and its load does not. Measured on the step-3 fixture with free ends: the outlet
  cell aggraded **0.053 m per activation** — 0.91 m in 26 — and its backwater lifted
  the whole reach's depth by 30%, while the inlet cell (exporting with no supply)
  scoured 1.12 m. This is not fixture-specific: `reach_alluvial.toml` has an open
  boundary and will do the same, and the sill is a *hydraulic* error long before it
  is a visible one. Either pin the end cells (`dz_lo == dz_hi == 0`, banked into the
  ledger) or keep every quoted measurement clear of the boundaries and say which.
- **A clamped step is not a free step: sync-point `dt` clamping degrades
  local-inertial, and no existing gate can see it.** Found at build step 8 while
  measuring §3's interval-independence gate, which is why that gate is worded the way
  it is. **This is not a morphology bug** — it reproduces with sediment never armed —
  and it predates M7: `solver/scheduler.py` clamps every step with
  `dt = min(dt, next_sync - t)`, the same algebra M1–M4 ran inline, and sync points
  include the output cadence, forcing breakpoints and slow-process activations.

  Water only, uniform steady reach (`validation.bedwave` with the bump removed),
  5835 s, **no sediment anywhere**:

  | sync cadence | clamped steps | interior depth | ripple | mass |
  |---|---|---|---|---|
  | none | 0 | 1.495871..1.495881 | 0.010 mm | 7.7e-09 |
  | 900 s | 7 | 1.495786..1.495951 | 0.165 mm | 7.8e-09 |
  | 300 s | 20 | 1.492152..1.501927 | 9.775 mm | 1.7e-08 |
  | 45 s | 130 | 1.489426..1.503675 | 14.248 mm | 2.5e-08 |
  | 22.5 s | 260 | 1.443363..1.517787 | 74.424 mm | 4.9e-09 |
  | 11.25 s | 519 | **0.230063..2.571629** | **2341.566 mm** | 9.5e-09 |

  **The mass gate reads 1e-8 throughout.** Mass is conserved perfectly; the water is
  simply in the wrong places, as a short-wavelength standing oscillation (unit
  discharge swinging 2.34..2.66 about a design 2.5).

  The mechanism is **local-inertial plus an abrupt Δt change**, isolated by two
  controls. Both per-step operations are exactly linear in `dt` by inspection (inflow
  adds `Q(t+dt/2)·dt`; the open sink removes `dt/dx·q_out`, capped at a depth the cap
  never reaches here), and the controls confirm they are not involved: (1) the same
  reach with **no sync cadence at all**, merely shortening one step in N to 0.25 Δt,
  reproduces it — 13.7 mm at 1 in 200, 2289 mm at 1 in 25; (2) a **closed box** with
  no inflow, no open edge and nothing per-step but the scheme reaches a 2015 mm ripple
  and a cell of exactly zero depth. Smooth drift of the state-derived Δt, which every
  run already has, does not do this; shorten-then-restore does. Capping the fast step
  so it divides the interval exactly — so the clamp never fires — removes it
  (±0.04 mm at 22.5 s), which is how it was told apart from the step size.

  **Why nothing caught it:** existing scenarios run a 900 s output cadence, which is
  0.165 mm here, and the validated benchmarks (dam-break, Manning normal depth to
  0.59–1%, the EA tests) all bound the reach against a reference that a large ripple
  would break. M7 is the first milestone wanting a *frequent* slow clock, because the
  morphological Courant number forces a short interval on an erosive reach — and
  morphology **rectifies** the oscillation into a permanent bed signature instead of
  letting it average out.

  **Candidate fix, measured, deliberately not shipped in step 8:** fill the span to
  the next sync point with `n = ceil(span/dt)` *equal* steps rather than full steps
  plus a remainder — still never exceeding the state-derived `dt`, still landing
  exactly on the sync point. Ripple 14.248 → **0.009 mm** at 45 s, 2341.566 → 1.908 mm
  at 11.25 s. It is not step 8's to ship: it rewrites the Δt sequence of *every* run,
  so the pre-M5 bitwise-identity invariant M4/M5/M6 all held — and the in-tree test
  that replays the pre-M5 inline loop as an executable reference — would move. That is
  a milestone-scale change deserving its own commit and its own before/after, the way
  the precision pass got one. Two numbers in it are also still unexplained: 22.5 s and
  11.25 s both settle at 1.908 mm, and a 32.9 mm end-cell offset is identical across
  all three cadences. Carried the way M6 carried `coarsen = 4`.

- **The morphological Courant number samples the flow; the transport integrates it.**
  `celerity_field` runs at the activation *instant*, so an interval in which a flood
  arrived, moved the bed and drained away reports Courant 0.000 — measured, 577 m³ of
  sediment moved at a reported zero, against 5.05 for the same scenario with its
  forcing held on. Treat a silent run with spiky forcing as unchecked rather than as
  cleared. A true interval maximum needs a full-field host reduction every fast step.

- **Every scenario still writes to the same default output**, and the frames
  export still never purges it. Pass `--out` / `--frames-dir` for anything worth
  keeping (CLAUDE.md's sharpest gotcha, and a morphology run is expensive enough
  to be worth not losing).

---

## 5. Open decisions for sign-off

1. **Demo scenario** — extend `reach_basin` with `[sediment]`, or a purpose-built
   alluvial reach where the transport is unambiguous? Recommendation: a new
   `scenarios/reach_alluvial.toml`, so the M6 demo stays a clean mosaic/channel
   regression and the morphology demo can be built to *show* something. Either
   way this is independent of the celerity scenario (build step 3), which is a
   gate fixture and is not up for the same decision.
2. **`interval_s` default** — 900 s follows the reservoir. Wants checking against
   the morphological CFL on the demo before it becomes a default. First datum from
   build step 3: 900 s puts the *celerity fixture* at Courant 3.1, so that fixture
   carries its own 45 s — a deliberately erosive reach at 2.5 m cells is the worst
   case, and it says the default cannot be assumed rather than that it is wrong.
   `reach_basin`-scale numbers (100 m cells, millimetre-per-activation channel
   transport) will land far under 1; the demo at step 9 decides it.
3. **Viewer** — confirmed deferred (§1.7), or is a static final-bed toggle wanted
   inside M7?
