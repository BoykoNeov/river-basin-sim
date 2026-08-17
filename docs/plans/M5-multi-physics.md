# M5 — Multi-physics (multi-rate scheduler, exercised by reservoir operations)

**Goal:** make the **multi-rate scheduler** real (HANDOFF §2, §8: *one simulated
clock; the flood kernel sub-cycles with a deterministic adaptive `Δt`; slow
processes advance at sync points via operator splitting*) and prove it with a slow
process that actually matters — **reservoir operations** (`[[structures]]` + release
rules). M5 also absorbs the two M4 deferrals that travel together: the
**`fixed_stage` boundary** and **EA SC080035 Test 1**, which needs `fixed_stage`
plus a **datum shift** (`z' = z − z_ref`) to survive float32 at its ~10 m datum.

Depends on: M4 (HLLC FV scheme, per-edge ghost BCs, conservative positivity
limiter, hardened mass gate — acceptance met 2026-07-02, confirmed by the user
before this milestone). Gate before M6 (reach / multi-resolution).

---

## 0. Scope — what M5 is and is *not*

**In (HANDOFF §9 M5, roadmap M5 line):**
- **Multi-rate scheduler** (`solver/scheduler.py`): a single simulated clock; the
  fast flood scheme sub-cycles with its own state-derived `Δt`; **slow processes**
  (reservoir release rules) run on their own coarse cadence at sync points, via
  operator splitting. The run loop's ad-hoc event clamping moves *into* the
  scheduler, unchanged in behaviour.
- **`[[structures]]` + release rules** (§7.1): a **dam** = a barrier (bed raised to
  the crest, so impoundment and overtopping are ordinary solver physics) plus a
  **controlled release** evaluated on the slow clock. Rules: `fixed` and
  `target_stage` (proportional draw-down). Mass-exact pool → outlet transfer.
- **`fixed_stage` boundary** (HANDOFF §8 BC list): a prescribed water-surface
  Dirichlet **ghost cell**, constant or piecewise-linear in time, per edge —
  **HLLC-only** (see §1.4). Mass banked through the edge in both directions.
- **Datum shift** `z' = z − z_ref`: an opt-in (`[grid] datum = "auto" | <float>`)
  elevation offset applied to the bed and to every absolute elevation in the config
  (stage levels, crest), so a domain at a high datum keeps float32 precision where
  it matters. The Zarr still stores the **true** bed; the shift is recorded.
- **EA SC080035 Test 1** (disconnected water body) as the validation gate for the
  above pair.

**Explicitly deferred (still a loud scope-gate error or a later milestone):**
- **`inflow` boundary *type*** — point-source `[[inflow]]` cells already cover
  prescribed discharge (M3) and are exact by construction; an edge-flux inflow BC
  adds no capability we lack. Stays a scope-gate error, now labelled *deferred*
  rather than assigned to a milestone.
- **Temporal rainfall** (`timeseries` / `storm_cells`) — unchanged, still later.
- **Multi-resolution / tiling-at-scale, sub-grid channels → M6.**
- **Sediment / morphology on the slow clock → M7.** M5 builds the scheduler seam
  sediment will plug into, and nothing more.
- **`fixed_stage` for the local-inertial scheme** — see §1.4; LI rejects it loudly.

---

## 1. Design decisions

### 1.1 The scheduler is a clock, not a driver
The temptation is to make `Scheduler` own the whole run (state, writer, ledger,
status). That inverts the existing structure for no gain and makes it untestable
without a GPU. Instead the scheduler owns **exactly the thing that is hard**: the
single simulated clock and the sync-point algebra. It yields `Tick`s; `run.py`
still does the stepping, forcing, accounting and I/O.

```python
sched = MultiRateScheduler(end_time=..., output_every=..., events=[...],
                           processes=[SlowProcess(name, interval, advance), ...])
for tick in sched.ticks(dt_fn):          # dt_fn() -> the scheme's state-derived dt
    ...                                  # inject forcing, scheme.step(tick.dt)
    for proc, elapsed in tick.due: ...   # operator-split slow update at tick.t1
    if tick.is_output: ...               # record + write
```

- **Sync points** are the union of: output times, forcing breakpoints (rain on/off,
  hydrograph knots), `end_time`, and every slow-process activation
  (`k * interval`). A fast step is clamped so it **never crosses** one. That is
  exactly what M1–M4's `_next_event_time` did — the slow activations are simply new
  members of the same set, which is why **existing runs stay bitwise-identical**
  (no structures ⇒ the same event list, the same arithmetic, in the same order).
- **Determinism (HANDOFF §8/§12):** activation times are `k * interval` computed
  from the simulated clock, never wall-clock, never step-count. `dt_fn` remains the
  scheme's state-derived CFL. A slow process advances by the **exact elapsed
  simulated time** since its last activation, so operator splitting is reproducible
  and independent of how many fast sub-steps happened in between.
- **Why yield instead of callbacks:** a generator makes the clock unit-testable with
  a stub `dt_fn` and no Warp state at all (pure arithmetic tests: clamping,
  ordering, no-crossing, exact activation times, float-tolerant termination).

### 1.2 A dam is geometry + a rule, not a new momentum term
Modelling a dam as a special face-level flux law would mean touching both schemes'
inner kernels — expensive, and a fidelity claim we cannot validate. Instead:

- **Barrier = bed geometry.** At setup, `z[cells] = max(z[cells], crest)`. Water
  impounds behind it; **overtopping is ordinary shallow-water physics** the
  validated scheme already handles (this is exactly how the EA Test 3 obstruction
  works). The modified bed is what gets written to the Zarr, so the viewer shows the
  structure.
- **Release = a slow-clock operator-split source/sink pair.** At each activation the
  rule reads the pool stage and returns a discharge `Q`; the process moves
  `V = Q · Δt_slow` from the pool region to the outlet cell in one splitting step.
  This is the textbook shape of the multi-rate design (§8) and it is what makes the
  scheduler load-bearing rather than decorative: the rule *only* sees state at sync
  points, and the flood scheme sub-cycles freely in between.

**Mass exactness.** The transfer is internal, so the ledger must see zero net
change. Achieved the same way M3's infiltration is exact: the withdrawal kernel
banks each cell's **actual float32 depth change** into a float64 accumulator; the
host sums it (a sync point, so the readback is cheap) and the delivery kernel adds
that volume at the outlet and banks the float32 rounding residual into `loss_cum`.
Removed volume − delivered volume = the banked residual, exactly, to the bit.

**Rules.** `fixed` (constant `release_m3_s`) and `target_stage` (proportional:
`Q = q_max · clamp((stage − target)/(crest − target), 0, 1)`) — one open-loop, one
closed-loop, which is what proves the sync-point feedback path.

> **Superseded during the build.** This plan specified the pool stage as `max(z + h)`
> over wet pool cells, and the outlet as a single cell. The demo scenario broke both
> (see *What the demo caught* in §5): the stage is now read at a **gauge cell — the
> pool's deepest point** — and the outlet is a **box**. The rest of §1.2 stands.

### 1.3 `fixed_stage` is a ghost cell, like the M4 BCs
M4 landed per-edge ghosts for closed (reflective) and open (transmissive). A
prescribed stage is the third member of the same family: the ghost gets
`η_ghost = stage(t)`, `h_ghost = max(stage − z_edge, 0)`, `z_ghost = z_edge` (so the
Audusse bed correction vanishes at the boundary face, as for the wall) and
**zero-gradient velocity**. The HLLC flux then decides the direction — a stage above
the interior surface drives inflow, below it drives outflow, and a *dry* boundary
(`stage ≤ z_edge`) is a no-op wall by construction.

- **Banking is unchanged and already signed.** M4's `_bank_*` kernels bank
  `−flux` as a loss; a stage edge that draws water *in* banks a negative loss, which
  the float64 ledger reads as inflow. No new accounting path.
- **Time-varying stage** is a piecewise-linear `[[t, level]]` curve evaluated
  **host-side at the step midpoint** and passed to the kernel as a scalar — a
  deterministic, allocation-free scalar per step, and second-order for a linear
  segment. Its knots join the scheduler's sync points, so a step never straddles a
  slope change in the boundary forcing.
- **The positivity limiter already covers it:** `_mass_beta` includes boundary faces,
  and `_limit_fx/_limit_fy` leave *inflow* faces unscaled — so a stage-driven inflow
  is not spuriously capped, and a stage-driven outflow cannot over-drain its cell.

### 1.4 `fixed_stage` is HLLC-only, and says so
The local-inertial scheme has no boundary faces at all: its BCs are a zeroed edge
face (closed) plus a *post-interior* self-capping sink (open), because the M1 donor
limiter never scales edge faces (see `boundaries.py`). A Dirichlet **surface** BC
has no honest expression in that structure — the nearest thing would be a
pressure-driven edge flux with no limiter protection, i.e. exactly the failure mode
that shape was designed to avoid. So: `scheme = "local_inertial"` + any
`fixed_stage` edge is a **hard, milestone-naming error at run setup**, not a silent
approximation. This is stated in the config docs and the scenarios README.

### 1.5 The datum shift is a pre-processing option, not a numerics change
`η = h + z` in float32 has an absolute resolution of about `ULP(z)`: at a 10 m datum
that is ~1e-6 m, at a 5000 m datum ~5e-4 m — which is a *depth-sized* error for the
thin sheets these benchmarks care about, and it shows up first as a lake-at-rest
that will not stay flat. Rather than promote fields to float64 (HANDOFF §2 says
don't), M5 shifts the origin:

- `[grid] datum = "auto"` ⇒ `z_ref = floor(min(bed))`; `datum = <float>` ⇒ that
  value; absent ⇒ no shift (existing scenarios bitwise-unchanged).
- The shift applies to the bed **and to every absolute elevation the config carries**
  (stage levels, structure crests) — one function, one place, so they cannot drift
  apart.
- Depth, velocity and all mass accounting are datum-independent. The Zarr `bed` is
  written **un-shifted** (true elevation) with `datum_shift_m` in `.zattrs`, so
  nothing downstream — analysis or viewer — has to know.

---

## 2. Build order (each step keeps `ruff` + `pytest` green; commit each)

1. **Plan doc** (this file).
2. **Multi-rate scheduler** (`solver/scheduler.py`) + `run.py` refactored onto it.
   Pure-arithmetic unit tests (no Warp state) plus a regression that an existing
   scenario is **bitwise-identical** to the pre-scheduler run.
3. **Datum shift** — config + run wiring; a lake-at-rest-at-altitude test that
   *fails without it* is the honest demonstration.
4. **`fixed_stage` BC** — HLLC ghost kernels, per-edge config (inline table), stage
   curves on `State`, banking, LI rejection. Tests: equilibrium with a still
   interior, inflow and outflow through a stage edge, mass gate.
5. **`[[structures]]` + reservoir operations** — config parsing, barrier application,
   `solver/processes/reservoir.py` as a `SlowProcess`, release series in `.zattrs`.
   Tests: barrier impounds; `fixed` and `target_stage` draw the pool down; transfer
   is mass-exact; the slow clock is honoured (activations at multiples of the
   interval, independent of the fast `Δt`).
6. **EA SC080035 Test 1** — the disconnected water body, at its real ~10 m datum, on
   a time-varying `fixed_stage` edge.
7. **Demo scenario + docs** — a reservoir scenario, `scenarios/README.md`,
   `CLAUDE.md` status, roadmap, this plan's acceptance section.

Viewer changes are **not** required (the store contract is unchanged; a dam appears
in the bed field, which the viewer already renders).

---

## 3. Validation plan (the credibility gates)

| Check | Type | Gate |
|---|---|---|
| **Scheduler clock algebra** | unit | steps never cross a sync point; activations at exact `k·interval`; run terminates at `end_time` |
| **Pre/post-scheduler equivalence** | regression | an existing scenario's depth field is **bitwise-identical** |
| **Lake-at-rest at a high datum** | analytical | with the shift, `max|u,v|` at the float32 floor; without it, measurably worse (recorded, not hidden) |
| **`fixed_stage` equilibrium** | analytical | a domain whose surface equals the prescribed stage stays at rest; mass gate holds |
| **`fixed_stage` fill / drain** | conservation | water enters/leaves through the edge, banked both ways, `rel_error < 1e-6` |
| **Reservoir transfer exactness** | conservation | pool loss == outlet gain to the bit (residual banked); closed domain ⇒ volume unchanged by a release |
| **`target_stage` feedback** | behavioural | pool drawn toward the target, release stops at/below it, `Q ≤ q_max` |
| **EA Test 1** | inter-model qualitative | the disconnected pond **retains** water after the boundary recedes; the connected part follows it down |
| **Global mass balance** | always-on | `rel_error < 1e-6`, every new gate |

**Honesty note on EA Test 1.** The SC080035 numbers pinned during M4 are not
reachable from this session. Test 1 is therefore built **faithful in form** — the
report's *structure* (a small domain at a ~10 m datum, two depressions separated by a
ridge, a boundary water level that rises above the ridge and then falls back) — with
every reconstructed number labelled as such in the test docstring, and the gate set
on the **published qualitative finding** (a water body that becomes disconnected
retains its water instead of draining with the boundary) rather than on invented
tolerances. Same policy M4 used for Tests 2 and 3.

---

## 4. Risks / watch-items

- **Silent behaviour change in the run loop.** The scheduler refactor touches the
  one loop every prior milestone depends on. Mitigation: bitwise regression against a
  stored pre-refactor field, and no change to the event-list construction order
  (float arithmetic is order-sensitive).
- **Operator-splitting error is real, and is the point.** A slow process that
  applies a whole interval's worth of transfer in one shot is first-order in
  `Δt_slow`. That is the documented design (HANDOFF §8), not a bug — but the
  interval must be small enough that the rule's feedback is meaningful. Default
  15 min; documented, configurable, and the `target_stage` test pins the behaviour.
- **A release that outruns the pool.** `Q · Δt_slow` can exceed the water present.
  The withdrawal is capped at what each cell holds (as infiltration is), and the
  *delivered* volume is whatever was actually removed — never the requested amount.
- **Stage BC + limiter interaction.** M4's carried limitation ("open-boundary
  banking is exact only while the limiter doesn't rescale the banked face") applies
  verbatim to the stage edge. The fill/drain gate is deliberately run into the
  drain-to-empty regime to exercise it.
- **Datum shift touching only *some* elevations.** A crest or stage left unshifted
  would be silently wrong by `z_ref`. Mitigation: one `apply_datum_shift(scenario)`
  that owns every absolute elevation, and a test that a shifted and unshifted run of
  the same scenario agree on depth.

---

## 5. Acceptance / demo — MET

- [x] **Multi-rate scheduler** (`solver/scheduler.py`): one simulated clock, the fast
      scheme sub-cycling with its own state-derived `Δt`, sync points it may not cross,
      slow processes advancing there by the exact elapsed simulated time. A **clock,
      not a driver** — `run.py` keeps state/stepping/forcing/accounting/IO, so the
      sync-point algebra is unit-tested with a stub `dt_fn` and no GPU at all.
- [x] **Pre-M5 runs bitwise-identical**: verified against stored LI / M3-sources / HLLC
      baselines, and guarded in-tree by a test that replays the pre-M5 inline loop as an
      executable reference and asserts float equality on every step.
      **Retired 2026-08-17** (`scheduler-equal-steps.md`). It did its job — it proved
      this extraction was inert — but the arithmetic it pinned turned out to be a
      defect: filling a span with full steps plus a remainder hands local-inertial a
      discontinuity of up to 585x between adjacent steps. The reference loop stays in
      `solver/test_scheduler.py` as a tombstone; four invariants replace the assertion.
- [x] **`[[structures]]`** dams and levees: barrier geometry (`cells` + `crest_m` raised
      into the bed, so impoundment and overtopping stay ordinary solver physics) plus
      `fixed` / `target_stage` release rules on the structure's own slow clock; release
      series recorded in `.zattrs` as `reservoir_releases`.
- [x] **Mass-exact transfer**: 300.000 m³ requested, 300.000 removed, 300.000 delivered,
      **0.0** banked residual; a release outrunning its reservoir takes exactly what is
      there; `target_stage` converges 3.76 → 3.002 m with Q easing 3.04 → 0.01 m³/s under
      its cap; activations are bit-identical at `dt_max` 5.0 and 1.0.
- [x] **`fixed_stage`** per-edge BC on HLLC, constant and time-varying, banked in both
      directions; rejected loudly on the local-inertial scheme. Equilibrium leaves a lake
      at rest (1.2e-5 m/s, nothing crosses); per-edge fill ×4 settles exactly at the
      prescribed level (mass 2.7e-7); a drain to 90% dry cells — deliberately into the
      positivity-limiter regime M4 flagged — still banks exactly (7.3e-8).
- [x] **Datum shift** (`[grid] datum`), with the lake-at-rest-at-altitude test that
      *fails without it*: 7.5e-6 m/s shifted versus 2.5e-5 at 10 m, 7.8e-4 at 500 m and
      1.2e-2 at 5000 m. Asserted as a ratio, so a future no-op shift cannot pass.
- [x] **EA SC080035 Test 1**: the far pond floods to 0.393 m and still holds 0.317 m
      after 5 h at the 10.10 m ridge crest, while the connected pond returns exactly to
      the 9.700 m boundary and ground above the peak never wets. Mass 6.9e-7.
- [x] **Demo scenario** `scenarios/reservoir_release.toml` (+ `scripts/make_reservoir_demo.py`):
      inflow hydrographs fill a valley reservoir to 0.52 Mm³ (stage 77.0 m, below the
      78 m crest), then the `target_stage` rule draws it back down on its 900 s clock —
      engaging at 75.1 m with 2.3 m³/s, peaking at 40.6 m³/s, then easing 36.6 → 30.8 →
      25.3 → 20.5 → 16.4 → 12.7 as the pool falls toward the 75 m target — while a tidal
      `fixed_stage` southern edge passes water both ways. Mass **4.6e-7** over 5 h;
      `h_max` 5.9 m; the dam is in the stored bed at 78.0 m.
- [x] `ruff` + `ruff format` clean; **189 tests green** (184 without `--extra geo`).
- [x] **Confirmed by the user 2026-08-09** — *out of order*: M6 was built, merged and
      signed off first, so this §9 checkpoint was carried open across M6 rather than
      gating it. See the sign-off run below.

### Sign-off run (2026-08-09) — confirmed on hardware, after M6

The figures above were produced by the authoring session pre-merge. M6 then refactored
`solver/io/` (mosaic, coarsening, grid loading) underneath this scenario, so the
confirm run is **also a regression check** on that path: `reservoir_release.toml` loads
its single tile through the mosaic loader now.

Re-run on this machine (RTX 5090, Warp 1.14.0 / CUDA 12.9), both backends, output kept
out of `data/results/` so nothing interleaves with another scenario's frames:

```
uv run python scripts/make_reservoir_demo.py
uv run python -m solver.run --config scenarios/reservoir_release.toml --device cuda:0 \
    --out <dir>/cuda.zarr --frames-dir <dir>/cuda_frames
uv run python -m solver.run --config scenarios/reservoir_release.toml --device cpu ...
```

**Hard gate — mass balance.** CPU **1.36e-07**, CUDA **3.15e-07**, both under the 1e-6
gate; a **1.8e-07 backend delta**, the same order as the residual itself, so CUDA
reduction order is not what sets it (the same conclusion M6's sign-off reached). 16
frames each, `h.min() == 0.0` throughout — the positivity limiter never went negative.

**Acceptance claims — all hold, on both backends.** The pool fills to **77.04 m**,
below the 78.0 m crest, and never overtops; the dam is in the stored bed at exactly
**78.0 m** across the line; the `target_stage` rule engages at **75.11 m with
2.27 m³/s**, peaks at **40.8 m³/s**, then eases monotonically over its 900 s clock —
39.1 → 36.6 → 33.6 → 30.8 → 28.1 → 25.3 → 22.8 → 20.5 → 18.3 → 16.3 → 14.4 → **12.7**
— as the stage falls **77.04 → 75.63 m** toward its 75 m target. The sequence quoted in
the acceptance bullet above is every *second* activation of this one, and reproduces to
three significant figures.

**Small differences from the pre-merge figures, stated rather than smoothed.** Peak
`h_max` is **6.05 m** here versus the recorded 5.9 m, and peak volume in the pool box is
**0.495 Mm³** versus the recorded 0.52 Mm³ (whole-domain peak is 0.63 Mm³, so the two
figures may not be measuring the same box). The recorded mass residual was 4.6e-07;
both backends here come in *below* that. These are the informational part of the
acceptance record, not the gate — and the release trajectory, which is the actual
closed-loop behaviour under test, matches. Note that this trajectory is feedback
sampled every 900 s: a float32-scale difference in pool stage at one activation shifts
that activation's Q and therefore every stage after it, so divergence in the tail is the
controller working, not a defect.

**Scope call: no viewer leg.** M5's acceptance list has no viewer item, the Zarr
contract is scheme-agnostic, M4 step 11 already rendered an HLLC store, and M6 signed
the tiled reader on hardware. Frames were exported (16, manifest written) but not opened
in Godot.

`ruff` + `ruff format` clean; **228 tests green** at sign-off (the 189 above is this
milestone's historical count)..

### HANDOFF divergences (small, deliberate)
§7.1's sketch is followed except in three places, none contradicting §2/§3/§8:

1. **`fixed_stage` is a per-edge table, not a bare string.** §7.1 lists it alongside
   `closed`/`open` in `default = "..."`. It cannot be: it carries a level (and possibly
   a curve). So `default` stays string-only and a stage edge is written
   `east = { type = "fixed_stage", level = 10.35 }`.
2. **The `inflow` boundary *type* is not built** and is no longer assigned a milestone.
   `[[inflow]]` cell sources (M3) already cover prescribed discharge and their mass
   accounting is exact by construction; an edge-flux inflow BC would add a second, less
   exact path to the same capability. It remains a loud config error.
3. **`release_rule = "fixed"` needs a `release_m3_s`.** §7.1's example gives the rule
   name without a discharge; a rule with no magnitude has no meaning, so it is required.

### What the demo caught (both fixed here, not deferred)
Building the demo scenario surfaced two defects that no unit test had:

1. **A point outlet is a shock.** Operator splitting hands over a whole interval's
   release at once: 60 m³/s × 900 s = 54,000 m³, which into a single 40 m cell is a
   **34 m instantaneous column**. `outlet` now accepts a **box** (a single cell is the
   1×1 case) and the release is spread evenly over it. The splitting error itself is
   the documented design (§4); its *delivery footprint* was not, and shouldn't be a
   trap for the next scenario author.
2. **The pool gauge was fooled by stranded water.** `pool_stage()` read `max(z + h)`
   over the pool box. Any realistic box spans the valley walls, so a millimetre of
   water stranded high on a wall reported a near-crest stage — and the draw-down rule
   kept releasing at full discharge from a reservoir that had already emptied. The
   gauge is now a **stilling well at the pool's deepest cell**, which is both more
   robust and what a real reservoir gauge measures.

### Carried limitations (state these honestly)
- **EA Test 1's spec is reconstructed, not pinned.** The SC080035 figures pinned during
  M4 were not reachable from this session. The test's *form* is faithful and its
  docstring separates what is pinned (purpose, qualitative result, ~10 m datum, small
  domain, rise-and-fall boundary) from what is reconstructed (domain size, cell size,
  bed elevations, ridge height, stage times and levels). The gate is the published
  qualitative finding plus the mass gate — never an invented tolerance. **Re-pin the
  numbers from the report before calling this a quantitative reproduction.**
- **M4's "Test 1 needs the datum shift" does not hold.** The test is parametrized over
  both datums and gates both: they agree on every reported depth to three decimals and
  both clear the mass gate. The shift is worth having, but it earns its keep at
  hundreds of metres, not ten.
- **The stage BC's CFL comes from the interior.** `compute_dt` does not see the ghost,
  so a stage set far above the interior surface is a dam-break at the boundary whose
  first step can be under-resolved. Not a stability hazard (the donor-β limiter caps
  outflow; an inflow only raises `h`, which the next `compute_dt` sees), but ramp a
  stage rather than stepping it.
- **Operator splitting is first-order in `Δt_slow`**, by design (HANDOFF §8). The
  15-minute default keeps the `target_stage` feedback meaningful; a much coarser clock
  will visibly lag the pool.
- **The `target_stage` rule is proportional, not a real operating policy.** It ramps
  linearly between target and crest and converges *toward* the target without
  overshoot. Real rule curves (seasonal storage, flood-control drawdown, minimum
  environmental flow) are a later addition; the seam for them is `Structure.discharge_at`.
- **Reservoir stage is a point gauge**, so a pool whose deepest cell dries while water
  remains elsewhere reads as empty. That is the intended failure direction (it stops
  releasing) and matches a physical gauge, but it is a modelling choice, not a truth.
