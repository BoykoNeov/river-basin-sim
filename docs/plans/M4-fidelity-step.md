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
11. **Example scenario(s)** — an `scheme = "hllc_fv"` scenario; regenerate a real
    `results.zarr` + frames; **confirm checkpoint** (mass gate + rendered PNG + a
    side-by-side LI-vs-HLLC on the same scenario).
12. **Docs** — CLAUDE.md status, roadmap, this plan's acceptance section; note any
    HANDOFF divergences.

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

## 6. Open scope decisions (confirm before build step 3)
1. **`fixed_stage` BC** — include in M4 (recommended, HLLC-only, non-gating) or defer?
2. **EA subset** — Test 1 + Test 2 as the gate (recommended), with Test 5 as stretch;
   or a different selection?

*(Acceptance section added on completion, per the M0–M3 pattern.)*
