# M3 — Real Scenarios

**Goal:** move past the "one uniform rain on a closed box" demo to configurable,
physically-varied scenarios. After M3 a scenario can carry **spatially-varying
parameter fields** (Manning roughness, infiltration), **inflow hydrographs**
(prescribed discharge entering at points over time), and **open boundaries**
(water leaves the domain) — all driven from the §7.1 TOML, all mass-accounted,
all reproducible via a recorded **command log / provenance** block.

Depends on: M2 (config-in/results-out loop, status.json, viewer stream — done).
Gate before M4 (fidelity / HLLC).

---

## 0. Scope — what M3 is and is *not*

**In (HANDOFF §9 M3, roadmap):**
- **Spatially-varying parameter fields**: `manning_n` and `infiltration` as either
  a scalar *or* a raster field aligned to the terrain tile.
- **Inflow hydrographs**: `[[inflow]]` point sources with a piecewise-linear
  discharge-vs-time curve (m³/s), injected mass-conservatively.
- **Open boundaries**: transmissive / free-outflow edges (per-edge or default),
  with the leaving volume tracked as outflow in the mass ledger.
- **Command log / provenance**: the resolved scenario + a content hash recorded
  into the Zarr `.zattrs` and a sidecar so a run is fully reproducible/shareable
  (HANDOFF §2 "Scenario = config + parameter fields + command log").

**Explicitly deferred (still a loud scope-gate error):**
- **HLLC FV (`scheme = "hllc_fv"`) → M4.**
- **`[[structures]]` (dams/levees) + release rules → M5.**
- **`fixed_stage` boundaries → M4** (needs the well-balanced scheme to be
  meaningful; reject with a milestone-naming error for now).
- **Temporal rainfall (`type = "timeseries"` / `"storm_cells"`) → later.** M3 adds
  *spatial* rain (`type = "field"`); time-varying rain stays rejected.
- **Multi-tile / tiling-at-scale → M6.** Fields must match the single demo tile's
  `(ny, nx)` exactly.

---

## 1. Design decisions (and where they diverge from HANDOFF)

### 1.1 Parameter-field format — `.r32`, not `.tif` (divergence, flagged)
HANDOFF §7.1's example writes `infiltration = "data/fields/infil.tif"`. We use
**raw little-endian float32 `.r32`** (row-major `(y, x)`, exact grid dims) as the
primary field format instead, because:
- it reuses the **M0 `.r32` tile convention** the solver already reads (no new
  code path, same orientation guarantees the terrain/viewer rely on);
- it keeps **rasterio/GDAL out of the solver run path** — that stack is the
  optional `geo` extra used by the offline pipeline, and the solver core must stay
  dependency-light and deterministic.
A `.tif` is still accepted *iff* rasterio is importable (resampled/clipped to the
grid), as an offline convenience — but `.r32` is the contract. The pipeline gains
a small `rasterize`/`fields` helper later; for M3 fields are authored as `.r32`.
> This is a deliberate, discussable divergence — the seam is a contract and
> `.r32` honours it more cleanly than `.tif`. Raise if you'd rather hard-require
> GeoTIFF.

### 1.2 Manning as a field
`State` carries `n` as a `(ny, nx)` float32 field (a scalar config broadcasts to a
uniform field). The face friction reads the **average of the two adjacent cells'**
`n`. For a uniform field `0.5*(n+n) == n` bit-for-bit in float32, so the dam-break
validation and the M1/M2 demos stay **bitwise-identical** — the field path is a
strict generalization, not a physics change.

### 1.3 Infiltration — constant-rate, capped, tracked
A **constant-rate** loss (Horton final rate / uniform capacity), *not* Green-Ampt
(deferred). Per step, each cell loses `min(f·Δt, h)` (never negative), where `f`
is the local infiltration rate (m/s, from mm/hr scalar or field). The removed
depth is accumulated into a per-cell `loss_cum` field and summed host-side in
float64 at output cadence → added to the ledger **outflow** (so the mass gate now
tests a real sink). Per-cell accumulation has one writer, so it stays
deterministic (no float atomics — §8/§12).

### 1.4 Rainfall field
`type = "field"` loads a `(ny, nx)` rate field (mm/hr). Continuity reads
`rain[i,j]`. Mass inflow per raining step is `area · Δt · Σ rain` with `Σ rain`
computed once host-side (analytic, deterministic) — generalizes the current
uniform `add_rain_step`.

### 1.5 Inflow hydrographs — point sources
`[[inflow]]` entries: a `cell = [i, j]` and a `hydrograph = [[t, Q], ...]`
(seconds, m³/s), piecewise-linear, zero-held outside its range. Each step injects
`h[cell] += Q(t)·Δt / area` and adds `Q(t)·Δt` to the ledger **inflow**. Step
sizes are clamped to hydrograph breakpoints (reusing the existing `events`
mechanism), so `Q` is linear across any step and the trapezoidal volume is exact.
Injecting as a **cell source** (not an edge-flux BC) keeps mass bookkeeping
trivial and lets inflow enter anywhere (a river mouth mid-domain), which is what
"inflow hydrograph" means in practice.

### 1.6 Open boundaries — transmissive (free-outflow)
Per-edge type in `[boundaries]` (`default` + optional `north/south/east/west`),
each `"closed"` (M2 default) or `"open"`. Open edges use **zero-gradient
extrapolation of the boundary-face discharge, clamped to outflow-only**: the edge
face copies the adjacent interior parallel face's `q` when that carries water
*out* of the domain, and stays 0 otherwise (no spurious inflow from a dry
exterior). This drains a filling basin, is stable within LI's regime, and the
leaving volume is exactly the boundary-face flux — accumulated into `loss_cum`
(same channel as infiltration) and tracked as ledger outflow.
> `fixed_stage` (prescribed water surface) is the physically-richer open BC but is
> only well-behaved with the well-balanced scheme — deferred to M4. M3 ships the
> transmissive one, which is what "open boundary" needs for routing water off the
> edge.

### 1.7 Command log / provenance
Record, into the Zarr `.zattrs` and a sidecar `<out>.provenance.json`: the source
TOML path, the fully-resolved `Scenario` (all parameters after manifest
inheritance), the sha256 of the source TOML and of every referenced field file,
and the solver version. This *is* the reproducibility story (§7.4): config +
field bytes + this record reproduce the run. (Live in-run edits at sync points are
an M5 concern — this is the static provenance half.)

---

## 2. Mass balance under M3 (the credibility gauge stays honest)

Residual `= inflow_cum − outflow_cum − (V(t) − V(0))`, still float64/Kahan.
- **inflow_cum** += rain (field-sum·Δt while raining) + Σ hydrograph `Q·Δt`.
- **outflow_cum** = `area · Σ loss_cum` (infiltration + open-boundary flux),
  read from the accumulator field at each record.
The gate `< 1e-6` is unchanged and now exercises sinks and sources on both sides
of the ledger — a stronger test than M1/M2's rain-into-a-box.

---

## 3. Build order (each step keeps `ruff` + `pytest` green; commit + push each)

1. **Plan doc** (this file) + task list.
2. **`config.py` + `test_config.py`** — extend `Scenario`; accept field paths,
   `[[inflow]]`, per-edge open boundaries, `rainfall.type="field"`; narrow the
   scope gate to only M4/M5-deferred features. Parse-only (no solver change yet).
3. **`io/fields.py` + `test_fields.py`** — `load_field(value, grid, base_dir)` →
   scalar-broadcast or `.r32` (optional `.tif`), dimension-checked.
4. **Manning field** — `State.n` field; face-average in the LI kernels; dam-break
   stays bitwise-identical (regression-verified).
5. **Infiltration + rain field** — `loss_cum` accumulator, capped infiltration
   kernel, rain-field continuity; ledger outflow from `loss_cum`; unit + mass
   tests.
6. **Inflow hydrographs** — `processes/inflow.py`, run-loop injection +
   event-clamping + ledger inflow; unit + mass tests.
7. **Open boundaries** — per-edge BC kernels, outflow into `loss_cum`; **tilted-
   channel steady-flow validation** (`validation/test_channel_flow.py`): constant
   inflow at the head, open outflow at the toe → hard mass gate + loose
   normal-depth check.
8. **Provenance** — `io/provenance.py` (hashes + resolved scenario) wired into
   `run.py`; test.
9. **Example scenario(s)** + `run.py`/CLI wiring end-to-end; regenerate a real M3
   `results.zarr` + frames; **confirm checkpoint** (mass gate + a rendered PNG).
10. **Docs** — CLAUDE.md status, roadmap, this plan's acceptance section.

Viewer changes are **not** required for M3 acceptance (M2's depth viewer already
renders any run); a scenario-setup UI is explicitly M3-deferred per the M2 plan.
The M3 demo is driven from configs + the validation harness.

---

## 4. Acceptance / demo (to be checked off)

- [ ] A scenario with a Manning field, an infiltration field, an inflow
      hydrograph, and an open outflow edge runs to completion, writes the Zarr +
      frames + provenance, mass gate `< 1e-6`.
- [ ] Scope gate still rejects `hllc_fv` (M4), `[[structures]]` (M5),
      `fixed_stage` (M4), temporal rainfall — each naming its milestone.
- [ ] Uniform-parameter runs (dam-break, M1/M2 demo) are **bitwise-unchanged** —
      the field paths are strict generalizations.
- [ ] `validation/test_channel_flow.py`: inflow≈outflow at steady state (hard mass
      gate) and depth within a loose band of Manning normal depth.
- [ ] Provenance record round-trips: hashes stable, resolved scenario complete.
- [ ] `ruff` + `ruff format` clean; `pytest` green (new field/infil/inflow/open/
      provenance/channel tests included).
- **Stop and confirm before M4.**

---

## 5. Risks / watch-items
- **Open-BC stability out of regime.** The steep M0 tile already pushes LI out of
  regime (M1 note); an open edge there can drain fast. The donor-cell outflow
  limiter (M1) still guards non-negativity; validate open BC on a *mild* channel,
  and keep the M0-tile demo's edges **closed** unless a gentle edge is chosen.
- **Field/grid misalignment.** `.r32` must match `(ny, nx)` exactly — a hard error,
  not a silent resample. `.tif` resampling (if used) is offline and logged.
- **Determinism.** All new sinks/sources accumulate per-cell (one writer) or
  analytically host-side — no nondeterministic float atomics enter the ledger.
- **Mass double-counting.** Boundary outflow is removed by `update_h` *and*
  recorded in `loss_cum`; the accumulator must mirror exactly what continuity
  removed (same Δt/dx factor) or the gate will flag it — which is the point.
