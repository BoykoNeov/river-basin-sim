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
closed-loop, which is what proves the sync-point feedback path. Pool stage is
`max(z + h)` over **wet** pool cells (a still pool gives exactly its level).

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

## 5. Acceptance / demo

- [ ] `solver/scheduler.py`: one simulated clock, fast sub-cycling, slow processes at
      sync points via operator splitting; `run.py` on it; existing runs bitwise-equal.
- [ ] `[[structures]]` dams with `fixed` / `target_stage` release rules, exact
      pool → outlet mass transfer, release series recorded in `.zattrs`.
- [ ] `fixed_stage` per-edge BC (constant + time-varying) on HLLC, banked both ways;
      rejected loudly on LI.
- [ ] Datum shift, with a lake-at-rest-at-altitude test demonstrating why.
- [ ] EA SC080035 Test 1 (faithful-in-form, qualitative gate + mass gate).
- [ ] A reservoir demo scenario; `ruff` + full `pytest` green.
- [ ] Stop and confirm before M6.
