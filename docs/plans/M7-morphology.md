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
4. **`state.py` + `sources.py`** — arm `dz_cum` (**f64**, cell-centred), `qs_int`
   + its compensation (**f32**, faces, §1.3), `z0` (the pristine initial bed);
   generalise the exact-accumulation helpers to take a target array; extend
   `field_memory_mb`. Assert the bitwise-unchanged invariant for every existing
   scenario with sediment unarmed.
5. **`solver/processes/morphology.py`** — the slow process, modelled on
   `reservoir.py`: constructed after `State`, exposes `as_slow_process()`,
   `advance(t, dt_slow)` returns a record, `series` lands in `.zattrs`. Plus the
   per-fast-step accumulation hook in `run.py` (beside the inflow injector, which
   is already wired that way).
6. **`SedimentLedger`** in `massbalance.py` or beside it — same shape as
   `MassLedger`, same Kahan f64 accumulators, same causal peak floor, its own
   relative gate. Masked-cell and clamped transport bank here.
7. **Store + export** — `bed_change` through `ZarrWriter`, §7.2 and §7.1 edits in
   HANDOFF, the manifest note, `scripts/plot_bed_change.py`.
8. **Validation** — §3, every gate.
9. **Demo** — `scenarios/reach_alluvial.toml`, GPU sign-off, HANDOFF + CLAUDE.md
   + roadmap updates.

---

## 3. Validation plan (the credibility gates)

- **Sediment mass conservation** — the new primary gate, mirroring the water
  gate's 1e-6: `(1−p)·Σ Δz·A + banked = net sediment in − out`, relative, with a
  causal peak floor. An erode-then-redeposit run nets to zero and would otherwise
  trip on denominator collapse — the same trap M4 hit and fixed for water.
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
  pattern — self-consistent, in-tree, and genuinely discriminating.
- **Interval independence.** Halve `interval_s`; the final bed must agree to a
  stated tolerance. This is what §1.3's time-integrated flux buys and the reason
  to build it that way.
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
- **The channel storage curve amplifies depth by `dx/w`**, so a channel cell's
  shear is much larger than a naive `h·S` on the cell mean. Getting this wrong
  in either direction is a factor-of-`dx/w` error — up to ~15× on this demo.
  It is the single most likely physics bug in the milestone.
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
   the morphological CFL on the demo before it becomes a default.
3. **Viewer** — confirmed deferred (§1.7), or is a static final-bed toggle wanted
   inside M7?
