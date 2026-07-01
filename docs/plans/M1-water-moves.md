# M1 — Water Moves

**Goal:** first "water moves." A local-inertial (Bates 2010) shallow-water solver
on a uniform tiled raster in Warp, driven by uniform rainfall with closed
boundaries, writing canonical Zarr, with a live float64/Kahan mass-balance
diagnostic. **Validated against dam-break.**

Depends on: M0 (conditioned + tiled `.r32` bed, proven end to end). Gate before M2.

---

## 0. Scope — what M1 is and is *not*

**In (from HANDOFF §9 M1):** local-inertial scheme · uniform rainfall · closed
BCs · Zarr output (§7.2) · live mass-balance · dam-break validation.

**Explicitly deferred (do not build now):**
- **§7.1 TOML config loader → M2.** M1 is driven by a *minimal in-code scenario*
  object plus a thin `run.py`. The dam-break cases are code fixtures. The
  `solver.run --config <toml>` contract lands with M2, when the loop closes.
- **§7.3 per-frame viewer tiles + `io/viewer_export.py` → M2.** M1 emits only the
  canonical Zarr store. Nothing renders in Godot this milestone.
- **§7.4 subprocess / `status.json` → M2.**
- **Multi-rate scheduler / operator splitting → M5.** M1 is single-rate: one
  simulated clock, one adaptive Δt sub-cycling the flood kernel.
- **HLLC FV, open/inflow/fixed-stage BCs, spatially-varying parameter fields →
  M3/M4.** M1 has closed BCs and scalar Manning n only.

**File subset built this milestone (paths per HANDOFF §6):**
```
solver/core/grid.py            # staggered grid geometry + index helpers
solver/core/state.py           # h, qx, qy (+ bed z) field container
solver/core/local_inertial.py  # the M1 Warp kernels
solver/core/friction.py        # Manning term (folded into the flux update)
solver/core/boundaries.py      # closed (reflective) BC
solver/core/massbalance.py     # host-side float64/Kahan accounting
solver/io/zarr_writer.py       # canonical §7.2 store
solver/run.py                  # in-code scenario -> run -> results.zarr
validation/test_dam_break.py   # wet-bed (gate) + dry-bed (diagnostic)
validation/analytical.py       # Stoker + Ritter reference solutions
```

---

## 1. The scheme (HANDOFF §8, local-inertial)

Per x-face flux update (y-analog):

```
q^{n+1} = ( q^n − g · h_flow · Δt · ∂(h+z)/∂x )
          / ( 1 + g · Δt · n² · |q^n| / h_flow^{7/3} )
```

then continuity per cell:

```
h^{n+1} = h^n + Δt · (Σ q_in − Σ q_out) / Δx  + R·Δt
```

- `h_flow` at a face = `max(η_L, η_R) − max(z_L, z_R)` where `η = h + z` is the
  water-surface elevation. This is the *depth available for flow across the face*,
  not a cell depth.
- Stable step from **state, not wall-clock** (determinism, §12):
  `Δt = α · Δx / √(g · h_max)`, `α ≈ 0.7`, `h_max` = current max depth. Recompute
  each step from a GPU **atomic-max** reduction (max is order-independent → stays
  deterministic even under atomics).
- `R` = rainfall rate (m/s) minus infiltration (M1: infiltration = 0), applied as
  a source in continuity.

### 1.1 Staggered grid + sign convention (pin this once — classic bug site)

`h`, `z`, `η` are **cell-centered**, shape `(Y, X)`.
`qx` lives on **x-faces**, shape `(Y, X+1)`: `qx[i, j]` is the flux on the face
between cell `(i, j−1)` and `(i, j)`.
`qy` lives on **y-faces**, shape `(Y+1, X)`: `qy[i, j]` is the flux between cell
`(i−1, j)` and `(i, j)`.

Sign convention: **positive qx flows +x (left→right), positive qy flows +y
(top→bottom).** Continuity for interior cell `(i, j)`:

```
h[i,j] += Δt/Δx · ( qx[i,j] − qx[i,j+1] + qy[i,j] − qy[i+1,j] ) + R·Δt
```

(flux *in* through the left/top face minus flux *out* through the right/bottom
face). Write this diagram as a docstring in `grid.py` / `local_inertial.py`.

### 1.2 Wetting/drying guard (the NaN source — §12)

The friction denominator `1 + gΔt·n²|q| / h_flow^{7/3}` blows up as
`h_flow → 0`. Rule: compute `h_flow` per face; **if `h_flow < h_dry` (1e-3 m),
set that face flux to 0** and skip the update. Cell `u,v` for output are 0 when
`h < h_dry`. Never divide by an unguarded `h`.

*Fallback noted, not built:* if the frictionless front oscillates, apply the
de Almeida & Bates (2013) q-weighting in the flux numerator. Out of M1 scope
unless the wet-bed gate needs it.

### 1.3 Kernel decomposition (Warp, device-agnostic)

Keep kernels pure and device-flagged so tests run on CPU (CI, no GPU) and the
real run on CUDA:
1. `compute_eta` — `η = h + z` (or fold into flux kernels).
2. `update_qx`, `update_qy` — face flux updates (friction folded in).
3. `apply_closed_bc` — zero the domain-edge faces (see §2).
4. `update_h` — continuity + rainfall source.
5. `reduce_hmax` — atomic-max for the next Δt.

One outer Python loop advances the clock: reduce `h_max` → derive Δt (clamped to
`dt_max`, and to not overshoot the next output time) → launch the flux/BC/
continuity kernels → advance `t`. Output at `output_every` cadence.

---

## 2. Boundaries (M1: closed only)

Closed / reflective: **zero normal flux on every domain-edge face.** With the
staggering above that means `qx[:, 0] = qx[:, X] = 0` and
`qy[0, :] = qy[Y, :] = 0` every step (a kernel or slice assignment). No ghost
cells needed for closed BCs. Mass can only enter via rainfall and can only leave
by… nothing — so for a rain-on-closed-basin run, `inflow − ΔV` must balance to
round-off. That is the tightest mass check and a good smoke test independent of
dam-break.

---

## 3. Mass balance (HANDOFF §8 — the credibility gauge)

**Host-side, float64, Kahan** — computed at output cadence, *not* on-GPU every
step. Rationale: an atomic *sum* over float32 is **not** order-deterministic, and
determinism is a locked invariant (§2, §12). So copy `h` back to host and sum in
float64 (Kahan) at each write.

Per output step track, in float64:
- `V(t) = Δx² · Σ h` — stored volume.
- `inflow_cum` — cumulative rainfall volume = `Σ (R · area_wet_or_all · Δt)`
  accumulated in the step loop (uniform rain over the domain → `R · Δx² · Ncells ·
  Δt`; Kahan-summed across steps).
- `outflow_cum` — 0 for M1 closed BCs (kept in the ledger for M3 open BCs).
- **Residual:** `E(t) = inflow_cum − outflow_cum − (V(t) − V(0))`.
- **Relative error:** `|E(t)| / max(inflow_cum, V(t), ε)`.

**Gate:** relative error must stay `< 1e-6` for every run. Exceedance is a
**failing test**, not a warning (§10). Store the series in the Zarr `.zattrs`
(`mass_balance_series`) and print the running value in `run.py`.

---

## 4. Canonical output (HANDOFF §7.2)

`results.zarr/` group, dims `(time, y, x)`:
- `time` (T,) simulated seconds
- `depth` (T, Y, X) f32 — `h`
- `u`, `v` (T, Y, X) f32 — cell-centered velocity, `u = (qx[:,:-1]+qx[:,1:])/2 /
  h`, guarded to 0 where `h < h_dry` (y-analog for v)
- `bed` (Y, X) f32 — static `z`
- `.zattrs`: `crs`, `dx`, `units`, `scheme="local_inertial"`, `run hash`,
  `mass_balance_series`

Chunk `(1, Y, X)` per timestep on the time axis (viewer-tile-aligned chunking is
an M2 concern once per-frame export exists). Reuse the smoke-test's
xarray→Zarr path.

---

## 5. Validation (HANDOFF §10) — two gates, different tolerances

Local-inertial drops advective acceleration; dam-break is a shock problem whose
Stoker/Ritter reference comes from the *full* SWE. So LI legitimately smooths the
front and is a few-% off on shock celerity — **that is expected physics, not a
bug.** Therefore two separate gates:

### 5a. Mass conservation — **hard gate, tight**
Relative mass error `< 1e-6` (§3) for every validation run and every scenario
run. This is conservative-by-construction (continuity is a flux divergence); the
Kahan/float64 ledger proves it stays that way. **A failing mass check fails CI.**

### 5b. Wave-shape vs analytical — enforced (wet-bed) + diagnostic (dry-bed)
Per the developer's decision, **write both fixtures**:

- **Wet-bed Stoker + small Manning n — the enforced gate.** Water both sides of
  the dam, small friction to damp frontal oscillation and avoid the worst wet/dry
  behaviour. 1-D setup on a flat bed (single row, closed side walls). Compare the
  computed depth profile to the Stoker solution at time T. **Loose tolerance:**
  front position within a few %, depth RMSE below a generous band (target
  ~5–10%, calibrate on first run), profile roughly monotone, **no NaNs**. Do
  *not* set sub-1% here — LI cannot meet it and shouldn't.
- **Dry-bed Ritter — reported, non-blocking diagnostic.** Dry downstream bed
  (exercises the wetting front, the classic NaN source). Run it, compute the same
  metrics, **print/record but do not fail** on wave-shape (it will be more
  smeared). Its mass-conservation gate (5a) *does* still apply — a dry-bed run
  that leaks mass or NaNs is a real failure.

`validation/analytical.py` implements Stoker (wet-bed) and Ritter (dry-bed)
closed forms. `validation/test_dam_break.py` runs both on the Warp **CPU** backend
(`device="cpu"` — CI has no GPU per CLAUDE.md), asserts 5a for both and 5b only
for wet-bed.

---

## 6. Build order (each step keeps `ruff` + `pytest` green)

1. **`state.py` + `grid.py`** — field container (`h, qx, qy, z` as Warp arrays,
   staggered shapes) + index/geometry helpers + the sign-convention docstring.
2. **`local_inertial.py` kernels** — `update_qx/qy` (friction folded), `update_h`,
   `reduce_hmax`; plus `boundaries.py` closed BC. Unit-test a single step on a
   tiny hand-checkable grid (CPU).
3. **Step loop + Δt-from-state** in a `solver/core/` driver function; verify a
   flat lake-at-rest-ish sanity (uniform depth, flat bed → stays put, mass
   exact). *(Not the M4 well-balanced test — just a no-blow-up sanity.)*
4. **`massbalance.py`** — host float64/Kahan ledger; wire into the loop; the
   rain-on-closed-basin mass smoke (§2/§3) passes `< 1e-6`.
5. **`zarr_writer.py`** — §7.2 store + `.zattrs` including the mass series.
6. **`run.py`** — minimal in-code scenario → run → `results.zarr`; runs on GPU
   using the real M0 `.r32` bed (`data/tiles/demo/tile_00_00.r32`, memory-mapped
   as `z`) with uniform rainfall. This is the "water moves on real terrain" demo.
7. **`validation/analytical.py` + `test_dam_break.py`** — both fixtures, gates per
   §5. Wet-bed gate enforced, dry-bed diagnostic, mass `< 1e-6` both.

---

## 7. Acceptance / demo — MET (2026-07-01)

- [x] Uniform rainfall on the M0 terrain tile runs to completion on the GPU
      (RTX 5090, sm_120) and writes a valid `results.zarr` (depth/u/v/bed/time +
      mass series). Final-frame mean depth = 0.025 m = exactly the 25 mm of rain
      that fell; runoff concentrates in channels (median 1.2 mm, p99 0.5 m, max
      8.9 m in the valley), no negative depths.
- [x] Rain-on-closed-basin mass-conservation relative error `< 1e-6` (demo run:
      **2.1e-08**).
- [x] Wet-bed Stoker dam-break (**enforced**): mass **2.5e-9**, wave-shape
      nRMSE 0.074, shock-front error 9.5% (LI lags the shock slightly, as
      expected for a non-shock scheme), no NaNs.
- [x] Dry-bed Ritter dam-break (**diagnostic**): mass 2.5e-9, no NaNs; wave-shape
      nRMSE 0.075 recorded non-blocking.
- [x] `ruff` + `ruff format` clean; `pytest` green (17 tests) on the CPU backend;
      smoke test still green.
- **Stop and confirm before M2** (the loop closes). ← we are here.

## 7a. Scope change from the plan: flux limiter built in M1

The plan (§1.2) deferred the mass-conservative flux limiter as an out-of-scope
fallback. It was **built in M1** because the only M0 terrain tile is the Great
Smoky Mtns (steep) -- local-inertial's worst case. Dry-start rain there sheets to
~1 mm everywhere; the `sqrt(g*h)` timestep bound is then far too large for the
thin-sheet velocities, and continuity over-drains cells ~1000x into large
negative depths and blows up (h_max ~160 m, mass error 1.5e-4 -- the gate
correctly flagged it). The dam-break correctness gate passed regardless (it is
in-regime), confirming the kernels are sound and the blow-up was purely LI out of
regime (HANDOFF scopes LI to "vast lowland floodplains").

To make "water moves on real terrain" demoable, `local_inertial.py` gained a
**per-cell donor limiter** (`compute_outflow_beta` + `limit_qx`/`limit_qy`):
each cell computes `beta = min(1, h / requested_outflow_depth)` and every face is
scaled by its single *donor* (upwind) cell's beta. Because the shared face array
is scaled once, both neighbours see the identical corrected flux -> **mass is
conserved exactly** and no cell can drain below zero. The limiter is inactive
(`beta == 1`) whenever no cell is over-drained, so it does not perturb the
dam-break validation. It has its own conservation test
(`test_flux_limiter_conserves_mass_and_forbids_negative_depth`). The de Almeida
q-weighting numerator fallback remains unbuilt (not needed).

**What the steep demo does and does not validate.** Mass conservation (2.1e-8)
and the spatial pattern (mean depth = rain input; runoff concentrates in channels)
are sound. But in steep, out-of-regime cells the limiter throttles `beta` well
below 1, so the **transient dynamics and velocity fields there are limiter-shaped,
not validated local-inertial hydraulics**. Acceptable for an M1 "water moves"
demo; it must **not** harden into a fidelity claim carried into M2+. The validated
LI physics is the in-regime dam-break, not the mountain runoff.

---

## 8. Open questions / notes for the developer

- **Δ dam-break tolerance band** is calibrated on the first real LI run (§5b) —
  I'll report the actual RMSE/front error and propose the enforced threshold then,
  rather than guess it now.
- **Handoff note (per "handoff is discussable"):** HANDOFF §9 pairs a non-shock
  scheme (LI) with a shock test (dam-break). This plan honours it but splits the
  gate — tight on mass, loose on wave shape — so a correct LI result isn't read as
  a failure. Flagging explicitly rather than silently reinterpreting.
- **License drift — fixed (2026-07-01):** `pyproject.toml` (`LicenseRef-BNCL-1.0`),
  `README.md`, and `docs/plans/M0-foundation.md` now all say BNCL-1.0, matching the
  `LICENSE` file.
