# M4 — Fidelity Step (well-balanced HLLC finite volume)

**Goal:** add a second, higher-fidelity flood scheme — a **well-balanced Godunov
finite-volume solver with an HLLC approximate Riemann flux** (HANDOFF §8) —
selectable per-scenario via `scheme = "hllc_fv"`, sitting **behind the same kernel
interface** as the M1 local-inertial (LI) scheme. LI is not replaced: per the
locked decision (HANDOFF §2) LI stays the permanent coverage scheme for lowland
floodplains; HLLC is the fidelity option for shocks, transcritical flow, and
well-balanced wet/dry behaviour. The two coexist by scheme selection.

Depends on: M3 (scenario system, sources/sinks, provenance, mass ledger — done and
**confirmed closed 2026-07-02**). Gate before M5 (multi-rate scheduler).

---

## 0. Scope — what M4 is and is *not*

**In (HANDOFF §8/§9 M4, roadmap):**
- **Well-balanced HLLC FV scheme** (`scheme = "hllc_fv"`): cell-centred conservative
  state `U = [h, hu, hv]`, MUSCL slope-limited reconstruction, **hydrostatic
  reconstruction** (Audusse 2004) for the bed-slope source, **HLLC** flux (Toro) at
  faces, **SSP-RK2** time integration, semi-implicit friction, clean wet/dry.
- **Scheme dispatch**: `run.py` selects the scheme from `scenario.scheme`; the LI
  path is bitwise-unchanged.
- **Mass-gate hardening**: causal peak-volume floor in `massbalance.py` so a
  drain-to-empty run cannot trip the gate by denominator collapse (prerequisite —
  the EA suite drains domains). Ships **first**, standalone.
- **Validation**: **lake-at-rest** (well-balancedness — the discriminating gate),
  **dam-break on HLLC** (analytical Stoker; must beat or match LI's nRMSE 0.074),
  and a **2–3 case subset of the UK EA 2D benchmark suite** (SC080035).

**Scope decisions to confirm (see §6):**
- **`fixed_stage` boundary** — M3 §0 parked it "→ M4, needs the well-balanced
  scheme to be meaningful." It *can* land now (ghost-cell BCs make it cheap). The
  roadmap M4 line does not require it. **Recommendation: include it** as an HLLC-only
  BC, but keep it out of the hard acceptance gate (nice-to-have, not gating).
- **EA suite subset** — the full 8-case suite is out of scope for one milestone.
  **Recommendation: Test 1 (disconnected water body) + Test 2 (floodplain depression
  filling)** as the M4 gate (both exercise wet/dry + mass conservation +
  well-balancedness — exactly what HLLC buys over LI); **Test 5 (valley flooding)**
  as a stretch/deferred.

**Explicitly deferred (still a loud scope-gate error or a later milestone):**
- **`[[structures]]` (dams/levees) + release rules → M5.**
- **Multi-rate scheduler → M5** (M4 keeps the single-rate flood loop).
- **Multi-tile / tiling-at-scale → M6.**
- **EA Tests 6–8 (flume dam break, 1D-2D linking, urban direct rainfall)** — beyond
  the M4 gate; revisit case-by-case in later milestones.

---

## 1. Design decisions

### 1.1 "Same kernel interface" = function-signature level, not shared state layout
LI is **staggered** (`qx (ny,nx+1)`, `qy (ny+1,nx)` discharge per unit width on
faces); HLLC is **cell-centred conservative** (`hu, hv (ny,nx)` momentum at
centres). These layouts are genuinely different and must **not** be forced
together. The shared interface is the function pair the run loop already calls:

```
compute_dt(state, ...) -> float          # scheme-owned CFL
step(state, dt, *forcing) -> None        # scheme-owned update, mutates state
```

- **Dispatch**: replace `run.py`'s static `from solver.core.local_inertial import
  compute_dt, step` with `scheme = get_scheme(scenario.scheme)` returning the scheme
  module (or a small object exposing `compute_dt`/`step`). The run loop stays
  scheme-agnostic. Config already parses `scheme`; M4 removes the `hllc_fv` scope
  gate and wires dispatch.
- **compute_dt differs per scheme anyway** — LI: `α·Δx/√(g·h_max)`; HLLC:
  `C·Δx/(|u|+√(gh))` (velocity-dependent). Scheme-owned `compute_dt` handles it; the
  event-clamping in `run.py` (output cadence, rain on/off, hydrograph breakpoints) is
  unchanged and wraps whichever `dt` the scheme returns.

### 1.2 Momentum is scheme-owned optional state
Add cell-centred `hu, hv` to `State` as **optional fields** armed by a setter —
exactly the existing idiom for `infil` / `rain` / `loss_cum` (`None` unless needed).
LI leaves them `None` and uses `qx/qy`; HLLC arms them and leaves `qx/qy` unused.
`State.from_bed` is unchanged (LI default); a `State.arm_hllc()` (or scheme init)
allocates `hu, hv` and any scratch (reconstructed face states, RK stage buffers).
Because the LI fields and code path are untouched, **dam-break / M1 / M2 / M3 LI runs
stay bitwise-identical** — the same non-regression guarantee M3 gave for fields.

### 1.3 Boundaries stay inside `step()`
LI already applies its BCs inside `step` (`apply_closed_bc`, `apply_open_outflow`).
HLLC owns its own BC application (transmissive / closed / optional `fixed_stage` via
**ghost cells**, the natural FV idiom — not LI's post-interior sink). The run loop
never sees the difference. `boundaries.py` gains HLLC ghost-cell helpers alongside
the existing LI ones; nothing existing is disturbed.

### 1.4 velocities_numpy branches by scheme
Output (§7.2) is cell-centred `u, v` regardless of scheme. LI reconstructs from
`qx/qy` (current code); HLLC is `hu/h`, `hv/h` guarded by `H_DRY`. Trivial branch on
whether `hu` is armed. The Zarr/viewer contract is scheme-agnostic — **no viewer
change for M4.**

### 1.5 Numerics (HANDOFF §8, references are the spec)
- **Reconstruction**: MUSCL with a slope limiter (minmod to start — TVD, robust at
  wet/dry; consider MC/van Leer later). Reconstruct on **water surface η = h + z** and
  depth so the well-balanced property holds, per Audusse.
- **Hydrostatic reconstruction** (Audusse et al. 2004): reconstruct interface depths
  from η and the higher of the two bed values so the bed-slope source and the pressure
  flux cancel exactly at rest → **lake-at-rest exact.** This is *the* property that
  makes M4 "well-balanced"; land it early and gate on it.
- **HLLC flux** (Toro): two-wave + contact restoration; the contact wave carries the
  transverse momentum `hv` (x-sweep) correctly. Wave-speed estimates (Einfeldt/Toro).
- **Time integration**: SSP-RK2 (Heun) — two flux evaluations per step.
- **Wet/dry**: threshold `H_DRY` (reuse `grid.H_DRY`, 1e-3 m); zero velocity below it;
  hydrostatic reconstruction suppresses spurious wet/dry front fluxes; guard depth
  non-negativity (the FV update is conservative, but reconstruction + friction need the
  clamp).
- **Friction**: semi-implicit Manning, reusing `friction.py` where possible (Manning
  `n` is already a per-cell `State.n` field from M3 — HLLC reads it cell-centred).
- **2D**: dimensional splitting (Strang or simple sweep) as the first cut — simplest
  correct 2D from a validated 1D solver; unsplit/genuinely-2D is a later refinement if
  a benchmark demands it.

---

## 2. Mass-gate hardening (PR #1 — lands before any HLLC)

The roadmap and `massbalance.py:111-118` flag this: in a **drain-to-empty** run with
no inflow, `abs(inflow)` and `abs(v)` both → 0, so a tiny *absolute* residual could
trip the relative gate via **denominator collapse** rather than physics. M1–M3 fully
draining tests keep the residual proportionally small so it doesn't bite today — but
the EA suite drains domains fully, so fix it first.

**Fix:** track a causal peak volume and floor the denominator with it:
```
peak_v = max(peak_v, v)   # updated each record; "causal" = only volume seen so far
denom  = max(abs(inflow), abs(v), peak_v, 1e-12)
```
**Safety argument (verify, don't assume):**
- `peak_v` only ever *raises* `denom` ⇒ `rel_error` only ever *decreases* ⇒ every
  gate-inequality test (`rel < 1e-6`) that passes today still passes.
- For monotonic-fill runs `peak_v == v` at each record ⇒ **bitwise-identical**
  reported `rel_error`; M1/M2/M3 filling-run numbers are unchanged.
- **The one check:** grep the tests for any assertion of an *exact* `rel_error` value
  at a *draining* timestep — those values shift (downward). Threshold assertions
  (`< gate`) are safe; an exact-equality at a drain step would need updating.

**New test:** a drain-to-empty gate test (rain a closed box, then open an edge / infiltrate
to zero) asserting the gate holds through the drain — the case the current denominator
would have mishandled and that HLLC's EA runs will hit.

Ships as its own commit/PR, green on `ruff` + `pytest`, before touching the scheme.

---

## 3. Build order (each step keeps `ruff` + `pytest` green; commit + push each)

1. **Plan doc** (this file) + confirm the two §6 scope decisions.
2. **Mass-gate hardening** (§2) + drain-to-empty test. Standalone. *(mergeable now)*
3. **Scheme dispatch scaffold** — `get_scheme(name)` in a small module (e.g.
   `solver/core/schemes.py`); `run.py` dispatches; LI path bitwise-unchanged (regression
   test). `hllc_fv` still errors "not yet implemented" (not the scope gate — a stub).
4. **1D HLLC core** — Riemann solver + MUSCL + hydrostatic reconstruction in 1D,
   validated against the existing `validation/analytical.py` Stoker/Ritter in 1D. Cheap
   confidence before 2D; pure-NumPy reference kernel is fine here.
5. **2D cell-centred update** (dimensional split) as Warp kernels; `State` optional
   `hu, hv` + scratch; `velocities_numpy` branch (§1.4).
6. **Hydrostatic reconstruction → lake-at-rest** — `validation/test_lake_at_rest.py`:
   flat η over an arbitrary (sloped, bumpy) bed stays flat with `max|u,v| ~ 0` (hard
   gate). **This is the M4 "well-balanced" acceptance keystone — get it green early.**
7. **Wet/dry + semi-implicit friction** — `H_DRY` handling, non-negativity guard,
   reuse `friction.py`; unit tests + a Manning normal-depth check on HLLC (parallels the
   M3 channel test, now on the fidelity scheme).
   - **Done.** `_friction` now shares `friction.manning_denominator` with LI (same
     Manning slope; algebraically identical). **Found and fixed a real
     well-balancedness bug the step-6 keystone missed:** the fully-wet lake-at-rest
     never exercised a wet/dry front, and MUSCL reconstruction across a shoreline
     (a dry neighbour injects a spurious water/bed slope into the minmod stencil) spun
     a smooth bowl up to ~20 m/s. Fix = drop to first-order at any cell adjacent to a
     dry cell (`hllc._dryfactor`), applied *identically* in the flux and source kernels
     so first-order Audusse's exact balance is preserved; fully-wet interiors are
     bitwise-unchanged (dam-break/step-6 unaffected). New discriminating gate
     `test_shoreline_lake_at_rest_on_bumpy_bed` (dry islands, 115 internal shorelines,
     stays at rest to the float32 floor ~1e-5, was ~20 m/s pre-fix); plus puddle,
     dry-bed Ritter, and friction-damping tests. **Manning normal-depth check
     deferred to step 9** (confirmed with the user): it needs a spatially-varying
     steady flow, which develops a boundary-driven drawdown under HLLC's
     transmissive-on-`eta` edges (a uniform-depth flow on a slope has non-uniform
     `eta`; extrapolating the ghost bed does not fix it). It lands with the step-9
     inflow/open ghost-cell BCs, exactly as the M3 channel test uses.
8. **SSP-RK2** time integration + **dam-break on HLLC** — parametrize
   `validation/test_dam_break.py` over scheme so it guards **both** LI and HLLC; HLLC
   nRMSE must match or beat LI's 0.074 and improve shock-front placement.
   - **Done.** SSP-RK2 (Heun predictor/corrector, `_rk_stage1`/`_rk_stage2`) already
     landed with the step-5/6 2D update — this step's incremental work is the
     **dam-break consolidation**: `test_dam_break.py` now dispatches through
     `schemes.get_scheme` and is parametrized over `{local_inertial, hllc_fv}` with
     per-scheme CFL α (0.7 / 0.45) and per-scheme shape bands (LI loose 0.10/0.15;
     HLLC tight 0.03/0.05 — a band LI cannot meet). One `MassLedger` gate (`<1e-6`)
     now guards **both** schemes, wet- and dry-bed; the old looser `1e-5` in
     `test_hllc_2d.py` was just conservative — HLLC actually lands at **8.0e-10**
     (wet) / **1.2e-8** (dry, through the `wp.max(h,0)` wetting-front clamp).
     **Results:** LI stays bitwise-identical (nRMSE 0.0740, front 0.0953, mass
     2.46e-9); HLLC beats it on shape *and* front (nRMSE **0.0076**, front
     **0.0101**). The redundant `test_wet_bed_dam_break_beats_li` was removed from
     `test_hllc_2d.py` (that file keeps lake-at-rest + determinism); the dam-break
     shock gate now lives only in the parametrized `test_dam_break.py`. 101 tests
     green.
9. **Ghost-cell BCs** — transmissive + closed for HLLC; **`fixed_stage`** if confirmed
   in scope (§6). Per-edge, mirroring the M3 `[boundaries]` config.
   - **Done.** HLLC now reads the full per-edge map from `state.boundaries` (added
     alongside `open_edges`; `State` defaults to an all-closed box so `from_bed`
     runs are walled without a config call). The interior flux kernels still compute
     every face transmissively (edge-clamped ghost); two **post-flux per-edge
     corrections** (not a halo/shape rewrite) then run inside `_eval_L` before the
     divergence: **closed** = a reflective-wall flux recomputed from an explicit
     ghost with the normal velocity negated (`hllc._wall_x_west/_east`,
     `_wall_y_north/_south`) — by antisymmetry the mass **and** transverse flux are
     exactly 0 and the normal-momentum flux is the wall pressure; **at rest `u=0` so
     it is identical to transmissive → lake-at-rest preserved by construction** (the
     wall only bites in motion). **Open** = transmissive + **mass banking**: each
     SSP-RK2 stage banks `0.5*dt*(F_boundary)/dx` (the Heun weight; `loss_cum` is a
     per-cell *depth*) into `state.loss_cum`, so the float64 mass ledger stays
     balanced when water actually leaves. **This banking is exact only while the
     `wp.max(h,0)` positivity clamp never fires** — true for steady flow, but a
     drain-to-empty run trips it (a known limitation carried to the EA cases, step
     10). `fixed_stage` deferred (§6, non-gating; needs a numeric per-edge config
     extension + re-opens float32 datum-sensitivity). **Manning normal-depth check
     (deferred from step 7) now lands** (see step 7): a steady head-inflow /
     open-toe channel on a moderately steep (transcritical, Fr~1.1) bed settles to
     the analytical wide-channel normal depth to **0.59%** across a dead-uniform
     interior — far tighter than LI's [0.5, 2.0] band, because HLLC carries the full
     momentum balance. New gates in `validation/test_hllc_boundaries.py`:
     per-edge open-drain (parametrized ×4, banking-sign + mass gate, clamp-free),
     closed-wall reflection (pile-up + velocity reversal + exact mass, with an
     all-open through-flow contrast), and the Manning channel. Existing HLLC suite
     unchanged: dam-break stays **bitwise-equal** in the scored region (nRMSE
     0.0076, front 0.0101 — waves never reach the walls, so reflective vs
     transmissive is inert there), lake-at-rest 8.5e-6. The frictionless moving-slab
     control in `test_hllc_wetdry` was switched to explicit open edges (its intent
     is to isolate friction; under the new default walls it would slosh). 107 tests
     green.
10. **EA benchmark subset** — `validation/test_ea_*.py` for the confirmed cases (§6);
    geometry + tolerances **pinned from SC080035 at implementation** (the EA cases are
    mostly inter-model comparisons, not analytical — assert against the published
    results envelope / qualitative pass criteria, plus the always-on mass gate).
    - **In progress.** *Prerequisite done* (commit 688b294): the memory-flagged blocker
      was a non-conservative positivity clamp — the HLLC scheme kept depth ≥ 0 with
      `wp.max(h, 0)` in the RK stages, which invents mass whenever it fires and breaks
      the mass ledger / open-boundary banking on any run that drains a cell to dry
      (a full drain-to-empty measured ~6.5e-2, ~5 orders over the gate). Fixed by
      porting LI's donor-cell β to the HLLC **mass** flux (`hllc._mass_beta` /
      `_limit_fx` / `_limit_fy`): each mass face is scaled by its upwind cell's
      `β = min(1, h/out_depth)`, so `h + dt·L ≥ 0` per RK stage (SSP ⇒ `h^{n+1} ≥ 0`),
      mass is conserved exactly (shared face scaled once, by its donor), and the
      banking reads the limited flux. Mass-only (momentum untouched) → lake-at-rest
      exact, in-regime dam-break bitwise. New gate
      `test_hllc_drains_to_empty_mass_conservative` drains a plane to 0.4% of V0 with
      ~96% of cells below `H_DRY` (the clamp regime) → mass 3.0e-8.
    - **EA Test 2 done** (commit b2fb267): `validation/test_ea_test2.py` — a faithful
      "flattened egg-box" (2000×2000 m, 4×4 ~0.5 m depressions on a 1:1500/1:3000
      slope), corner inflow hydrograph (20 m³/s, ~85 min), closed walls, dry start.
      CI-tractable **40 m / 12 h** (report is 20 m / 48 h). Qualitative + mass gates
      per §5: mass 2.9e-7 (closed ⇒ stored = injected exactly), non-negative through
      wetting/drying, depressions fill (11/16 wet, deepest SE 0.28 m), and the
      up-slope **top-right (NE) depressions stay dry** (the report's points-15/16
      finding). The full 20 m / 48 h run is deferred to a step-11 GPU demo.
    - **EA Test 3 done — but reframed** (user picked Test 3 as the second case):
      `validation/test_ea_test3.py`. Faithful 300×100 m / 5 m / 15 min (the report's
      *exact* resolution + horizon — already CI-tractable), 1:200 slope, two Gaussian
      depressions at x=150/250, inflow line on the west, closed walls, dry start.
      **The "HLLC-vs-LI discriminator" premise was wrong** (advisor-corrected): report
      p.105 splits *zero-inertia* packages (which don't overtop) from *with-inertia*
      packages (which do), and **Bates local-inertial keeps `∂q/∂t`**, so LI is a
      with-inertia scheme on the *same* side as HLLC — empirically **both overtop**.
      So the gate is a **within-HLLC momentum-conservation test** instead: the
      obstruction crest is fixed by an offline **scheme-free volume anchor**
      (depression-1 capacity integrated to the crest; inflow = 0.9× it, so the static
      equilibrium sits below the crest — no static spillover), then two HLLC runs
      inject the **same volume** differing only in hydrograph sharpness. Gentle
      (tb=300 s) → null: P1 settles 3.7 cm below crest, P2 **dry** (settled, P2 spread
      0 over the last 300 s). Sharp (tb=80 s, same volume) → P1 still below crest but
      momentum carries a splash over → P2 **wets to 6 cm** (≈ the report's ~5–6 cm).
      Only arrival momentum differs → isolates the momentum transport Test 3 targets.
      Crest never tuned to outcome; only hydrograph energy differs, at fixed volume.
      A second (non-gating) context test records that **both** LI and HLLC overtop the
      sharp pulse — the honest result explaining why HLLC-vs-LI is not a discriminator
      here. Mass 2.1e-9 (null) / 6.7e-8 (signal); h ≥ 0; **111 tests green**.
      ⚠️ **The claim changed** — this is a momentum-conservation reproduction of Test 3,
      **not** a scheme discriminator. **User signed off (2026-07-02, option (a)):** the
      reframe satisfies Test 3 for M4 acceptance; Test 2 + Test 3-as-momentum-gate close
      the EA subset. Test 1 (needs `fixed_stage` + datum-shift) stays a post-M4 addition,
      not a step-10 blocker. **Step 10 CLOSED.**
    - *(superseded fork, for the record)* §6's pick was Test 1 + Test 2, but **Test 1
      (disconnected water body) requires a time-varying water-level (`fixed_stage`)
      boundary** — the Dirichlet ghost deferred in step 9 (§6 open decision #1) — and
      its ~10 m datum makes the honest route `fixed_stage` **plus** a datum-shift
      (`z' = z − z_ref`). Test 3 was chosen as the lower-effort second case; if the
      reframe above is unsatisfying, (a) Test 1 (build `fixed_stage` + datum-shift) or
      (c) gate M4 on Test 2 alone remain open. Specs pinned in the EA-benchmarks memory.
11. **Example scenario(s)** — an `scheme = "hllc_fv"` scenario; regenerate a real
    `results.zarr` + frames; **confirm checkpoint** (mass gate + rendered PNG + a
    side-by-side LI-vs-HLLC on the same scenario).
    - **Done** (commit 67eea13). `scenarios/river_reach_hllc.toml` is
      `river_reach.toml` with **only** `[meta].scheme` and the CFL changed (0.45 vs
      LI's 0.70 — HLLC's bound is velocity-dependent), so the pair is a like-for-like
      side-by-side on the same real tile. Verified end to end on the RTX 5090 (13
      frames, 1 h sim): HLLC **6.66e-7** < gate, LI baseline **1.24e-7**. The fields
      agree at the gross level (wet-depth r = 0.965, total volumes within 4.3%) with
      localized front-placement differences, and the donor-β positivity limiter fires
      on **0 deep channel cells at every frame** — so the channels are out of limiter
      regime and the comparison reflects real scheme hydraulics, not clamp-vs-clamp.
      Godot renders the HLLC store unchanged (scheme-agnostic Zarr contract).
12. **Docs** — CLAUDE.md status, roadmap, this plan's acceptance section; note any
    HANDOFF divergences.
    - **Done.** CLAUDE.md status gains an M4 entry (M3 flipped to done), roadmap M4
      → acceptance met, §4/§6 below record the resolved scope decisions and the
      carried limitations. **No HANDOFF divergences**: §8's HLLC/well-balanced/wet-dry
      spec is implemented as written; the two departures from *this plan* are the §6
      decisions (`fixed_stage` deferred, EA Test 3 substituted for Test 1), both
      user-confirmed and neither contradicting HANDOFF.

Viewer changes are **not** required for M4 (the depth/velocity viewer renders any run;
the store contract is scheme-agnostic).

---

## 4. Validation plan (the credibility gates)

| Check | Type | Gate |
|---|---|---|
| **Lake-at-rest** | analytical (stays flat) | `max|u,v|` below tol on arbitrary bed — **hard, discriminating** |
| **Dam-break (HLLC)** | analytical Stoker | nRMSE ≤ LI's 0.074; front placement improved |
| **Manning normal depth (HLLC)** | analytical | within ~1% (parallels M3 channel) — **done (step 9)**: 0.59% on a transcritical channel, `validation/test_hllc_boundaries.py` |
| **HLLC closed wall / per-edge open drain** | reflection + mass | wall reflects & conserves mass; each open edge drains with the gate holding (step 9) |
| **Shoreline lake-at-rest (bumpy bed, dry islands)** | analytical (stays flat) | `max|u,v|` at float32 floor — **hard, discriminating** wet/dry well-balancedness (step 7) |
| **EA Test 1 / Test 2** | inter-model envelope | qualitative pass vs SC080035 published results |
| **Global mass balance** | always-on | `rel_error < 1e-6` (now peak-floored, §2) |

Dam-break stays green on **LI** too (parametrized) — HLLC must not regress the
coverage scheme.

---

## 5. Risks / watch-items
- **Biggest numerics lift in the project.** Slice per §3; each step is independently
  testable. Don't write 2D-unsplit-HLLC-with-friction in one go.
- **Wetting/drying instability** (HANDOFF §12, the classic NaN source). Hydrostatic
  reconstruction + clean `H_DRY` + non-negativity guard are the defence; test wet/dry
  fronts explicitly (EA Test 1/2 are exactly this).
- **Well-balancedness is easy to *almost* get.** A scheme that's off by O(machine-eps)
  at rest still "looks" flat but drifts on long runs — gate lake-at-rest tightly and on
  a genuinely bumpy bed, not a flat one.
- **EA tolerances are not analytical.** The EA cases compare against an envelope of
  commercial-package results, not a closed form. Pin the pass criteria from the report;
  don't invent an nRMSE the report doesn't define.
- **Determinism** — HLLC's `compute_dt` reads a velocity-based max; use the same
  order-independent atomic-max reduction pattern LI uses (`reduce_hmax`), extended to
  `|u|+√(gh)`, so `dt` stays state-derived and reproducible (HANDOFF §8/§12).
- **float32 at fronts.** HLLC wave speeds and star states in float32 can lose precision
  near dry states; the float64/Kahan mass gate is the guard (do not relax it), and clamp
  small depths before dividing.

---

## 6. Open scope decisions — RESOLVED
1. **`fixed_stage` BC** — **deferred past M4** (the recommendation was to include it;
   it was not gating, and step 9 landed transmissive + closed ghosts without it). It
   needs a numeric per-edge config extension *and* re-opens float32 datum sensitivity
   (EA Test 1's ~10 m datum wants a `z' = z − z_ref` shift alongside it). `config.py`
   still rejects it with the milestone-naming scope-gate error — that message now
   points one milestone late and is corrected as part of M5's intake.
2. **EA subset** — landed as **Test 2 + Test 3**, not the recommended Test 1 + Test 2.
   Test 1 requires the deferred `fixed_stage` (decision 1), so Test 3 was substituted
   as the lower-effort second case; **user signed off 2026-07-02 (option a)**. Test 1
   and Test 5 are post-M4 additions. See the step-10 notes for the Test 3 reframe —
   it is a **within-HLLC momentum gate, not a scheme discriminator**.

---

## 7. Acceptance / demo — MET (2026-07-02)

- [x] **Well-balanced HLLC FV scheme** (`scheme = "hllc_fv"`): MUSCL/minmod
      reconstruction on η, **hydrostatic reconstruction** (Audusse 2004), HLLC flux
      with contact restoration, **SSP-RK2**, semi-implicit Manning sharing
      `friction.manning_denominator` with LI, clean `H_DRY` wet/dry.
- [x] **Same kernel interface** (§1.1): `schemes.get_scheme(name)` returns the module
      providing `compute_dt`/`step`; the run loop never branches on scheme. Momentum
      `hu/hv` is optional armed state (§1.2), so the **LI path is bitwise-unchanged** —
      LI dam-break still reports nRMSE 0.0740 / front 0.0953 / mass 2.46e-9.
- [x] **Lake-at-rest** (the well-balancedness keystone): flat η over a sloped, bumpy
      bed stays flat — **8.5e-6**. And the *discriminating* wet/dry version,
      `test_shoreline_lake_at_rest_on_bumpy_bed` (dry islands, 115 internal
      shorelines), holds at the **float32 floor ~1e-5** — this caught a real bug
      (MUSCL across a shoreline spun a bowl to ~20 m/s; fixed by dropping to first
      order adjacent to dry cells, identically in the flux *and* source kernels).
- [x] **Dam-break on HLLC** beats LI on shape *and* front: nRMSE **0.0076** (vs
      0.0740) and front **0.0101** (vs 0.0953), against analytical Stoker. One
      parametrized `test_dam_break.py` now gates **both** schemes, wet- and dry-bed,
      under a single `<1e-6` mass gate (HLLC: 8.0e-10 wet, 1.2e-8 dry).
- [x] **Manning normal depth (HLLC)**: a steady head-inflow / open-toe transcritical
      channel (Fr ≈ 1.1) settles to the analytical wide-channel normal depth to
      **0.59%** across a dead-uniform interior — far tighter than LI's [0.5, 2.0] band.
- [x] **Ghost-cell BCs** (step 9): per-edge closed (reflective wall, exactly
      antisymmetric ⇒ identical to transmissive at rest ⇒ lake-at-rest preserved by
      construction) + open (transmissive with per-RK-stage mass banking into the f64
      ledger). Gated per-edge ×4 for drain, plus wall reflection + through-flow contrast.
- [x] **Mass-gate hardening** (§2): causal peak-volume floor in `massbalance.py`, so a
      drain-to-empty run can't trip the gate by denominator collapse. Monotonic-fill
      runs are bitwise-unchanged (`peak_v == v`).
- [x] **Conservative positivity limiter**: the old `wp.max(h, 0)` clamp invented mass
      whenever it fired (a full drain measured ~6.5e-2, five orders over the gate).
      Replaced by LI's donor-cell β on the HLLC **mass** flux — non-negativity per RK
      stage, exact conservation, momentum untouched (lake-at-rest exact, in-regime
      dam-break bitwise). `test_hllc_drains_to_empty_mass_conservative`: drains to 0.4%
      of V0 with ~96% of cells below `H_DRY` → **3.0e-8**.
- [x] **EA SC080035 Test 2** (floodplain depression filling): faithful flattened
      egg-box, corner hydrograph, dry start, at CI-tractable **40 m / 12 h** (report:
      20 m / 48 h). Mass **2.9e-7**; 11/16 depressions fill; the up-slope NE
      depressions **stay dry**, reproducing the report's points-15/16 finding.
- [x] **EA SC080035 Test 3** (momentum over an obstruction): the report's *exact*
      300×100 m / 5 m / 15 min setup. Gate is **within-HLLC**: crest fixed by an
      offline scheme-free volume anchor, then two runs inject the **same volume**
      differing only in hydrograph sharpness — gentle → P2 dry (null), sharp → P2 wets
      to 6 cm (≈ the report's 5–6 cm). Mass 2.1e-9 / 6.7e-8.
- [x] **GPU demo + side-by-side**: `scenarios/river_reach_hllc.toml` on the real Smoky
      Mtns tile, mass **6.66e-7** (LI baseline 1.24e-7); fields agree grossly
      (r = 0.965, volumes within 4.3%) with the limiter firing on **0** channel cells.
- [x] `ruff` + `ruff format` clean; **111 tests green** (106 without `--extra geo`).
- **Stop and confirm before M5.** ← we are here.

### Carried limitations (state these honestly)
- **EA Test 3 is not an HLLC-vs-LI discriminator.** The original premise was wrong:
  report p.105 splits *zero-inertia* from *with-inertia* packages, and Bates LI keeps
  `∂q/∂t`, so LI sits on the **same** side as HLLC — empirically both overtop. A
  second, non-gating test records exactly that. Test 3 here is a momentum-conservation
  reproduction.
- **Open-boundary mass banking is exact only while the positivity limiter doesn't
  rescale the banked face.** The donor-β limiter made drain-to-empty conservative
  (3.0e-8), but the banking/limiter interaction is the sharp edge in this scheme —
  re-check it before trusting a new open-boundary regime.
- **EA Test 2 runs at reduced resolution/horizon** (40 m / 12 h vs the report's
  20 m / 48 h) to stay CI-tractable. The full-fidelity run is a GPU demo, not a test.
- **`fixed_stage` and EA Test 1 are not in M4** (§6.1). The `config.py` scope-gate
  message for `fixed_stage` still says "arrive in M4" and needs re-pointing.
- The steep-tile caveat from M1 is **unchanged for LI**; HLLC now carries the full
  momentum balance there, but the M0 tile still has no measured-truth comparison —
  agreement between the two schemes is a consistency check, not a validation.
