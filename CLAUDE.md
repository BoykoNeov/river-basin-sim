# CLAUDE.md — River Basin Simulator

Project conventions for Claude Code. **`HANDOFF.md` is the source of truth** for
architecture, the locked decisions, the numerics spec, the component contracts,
and the milestone build order. Read it before non-trivial work; this file only
adds conventions and records what's easy to get wrong.

## What this is
A batch (not real-time) GPU shallow-water river-basin simulator. Three components
connected by **files, not code** (HANDOFF §4): a Python+Warp solver, an offline
data-prep pipeline, and a Godot 4 viewer. The seam is a contract (§7) — either
side can be rewritten as long as the formats hold.

A faithful research/education sandbox validated against benchmarks — **not a
regulatory-certification tool**. State that honestly anywhere it matters.

## Status
- **M4 — Fidelity step: acceptance met; confirm before M5.** A second, higher-fidelity
  scheme now coexists with LI by **selection, not replacement** (HANDOFF §2 — LI stays
  the permanent coverage scheme): a **well-balanced Godunov HLLC finite-volume solver**
  (`solver/core/hllc.py`, `scheme = "hllc_fv"`) — MUSCL/minmod on η, **hydrostatic
  reconstruction** (Audusse 2004), HLLC flux with contact restoration, **SSP-RK2**,
  semi-implicit Manning sharing `friction.manning_denominator` with LI. Dispatch is
  `solver/core/schemes.py::get_scheme` returning the module that owns
  `compute_dt`/`step`; momentum `hu/hv` is **optional armed state**, so the **LI path
  is bitwise-unchanged** (M1/M2/M3 runs identical). HLLC owns its BCs as **per-edge
  ghost cells** (closed = reflective wall, exactly antisymmetric ⇒ identical to
  transmissive at rest; open = transmissive + per-RK-stage mass banking into the f64
  ledger). Two mass-ledger hardenings: a **causal peak-volume floor** in
  `massbalance.py` (a drain-to-empty run can no longer trip the gate by denominator
  collapse) and a **conservative donor-cell β positivity limiter** on the HLLC mass
  flux, replacing a `wp.max(h,0)` clamp that invented mass whenever it fired (~6.5e-2
  on a full drain, five orders over the gate). **Validated:** lake-at-rest 8.5e-6 and
  — the discriminating one — **shoreline** lake-at-rest on a bumpy bed with dry
  islands at the float32 floor ~1e-5 (this caught a real bug: MUSCL across a shoreline
  spun a bowl to ~20 m/s); **dam-break beats LI on shape *and* front** (nRMSE 0.0076
  vs 0.0740, front 0.0101 vs 0.0953) under one parametrized test gating both schemes;
  **Manning normal depth 0.59%** on a transcritical channel; drain-to-empty 3.0e-8;
  **UK EA SC080035 Test 2 + Test 3**. GPU demo `scenarios/river_reach_hllc.toml` mass
  6.66e-7 vs the LI baseline's 1.24e-7 on the same scenario. No viewer change — the
  Zarr contract is scheme-agnostic. 111 tests green. **Read the plan's carried
  limitations before extending this**: EA Test 3 is a *within-HLLC momentum* gate, not
  a scheme discriminator (Bates LI keeps `∂q/∂t`, so it is on HLLC's side and both
  overtop); open-boundary banking is exact only while the limiter doesn't rescale the
  banked face; EA Test 2 runs at 40 m / 12 h vs the report's 20 m / 48 h. `fixed_stage`
  + EA Test 1 deferred past M4. See `docs/plans/M4-fidelity-step.md`.
- **M3 — Real scenarios: done.** The scenario system
  now carries real physics beyond "uniform rain on a closed box", all
  mass-accounted and reproducible: **spatially-varying parameter fields** (Manning
  + infiltration as scalar *or* `.r32` field via `solver/io/fields.py`; Manning is
  a per-cell `State.n` face-averaged in the LI kernels — uniform `n` is bitwise
  identical, so dam-break/M2 are unchanged), a **constant-rate infiltration sink**
  (capped, banks exact `f64` loss into `State.loss_cum`), **spatial rainfall
  fields**, **inflow hydrographs** (`solver/processes/inflow.py` — piecewise-linear
  `Q(t)` cell sources, event-clamped), and **open / free-outflow boundaries**
  (`solver/core/boundaries.py` — a *post-interior self-capping sink*, not a live
  boundary face, because the M1 limiter never scales edge faces; `loss_cum` is
  **float64** since open outflow concentrates at one edge cell). **Command-log /
  provenance** (`solver/io/provenance.py`) records source+field sha256 + resolved
  scenario into `.zattrs` + a `<store>.provenance.json` sidecar. **Validated:** a
  mild steady channel reaches **Manning normal depth within 1%**; a steep basin
  drains with `h.min() >= 0`. Two GPU demos green (`river_reach` mass 1.24e-7,
  `spatial_fields` 7.57e-8). 82 tests green. See `docs/plans/M3-real-scenarios.md`.
- **M2 — The loop closes: done.** The §7 contracts
  are live end to end: **§7.1 config-in** (`solver/io/config.py` — TOML → `Scenario`,
  parses the full schema but *rejects* not-yet-built features with a milestone-naming
  `ConfigError`, the scope gate; M4 opened `scheme = "hllc_fv"`, so what remains gated
  is `[[structures]]`, `fixed_stage`/`inflow` boundary types, and temporal rainfall),
  **§7.4 status.json** (`solver/io/status.py` —
  atomic `starting→running→writing→done|error`; wall-clock `eta_s` never touches
  the sim), **§7.3 per-frame viewer stream** (`solver/io/viewer_export.py` — a
  post-process over the Zarr: raw LE-f32 depth tiles + `manifest.json` with a
  *global* robust colormap range, p99 clamp so the thin sheet stays visible). The
  **Godot viewer** (`viewer/scenes/results_view.tscn`) launches the solver as a
  non-blocking subprocess (Windows batch + `uv run`), polls status at 4 Hz,
  auto-loads results, and renders a **lifted depth-coloured water surface**
  (`water_surface.gdshader`: η = bed + depth reconstructed in-shader, dry-cell
  discard) with a timeline scrubber. Full loop verified from Godot (`--rblaunch`)
  on the RTX 5090; mass gate 2.12e-8; error path writes `state="error"` (viewer
  never hangs); 38 tests green. See `docs/plans/M2-loop-closes.md`.
- **M1 — Water moves: done.** Local-inertial (Bates
  2010) shallow-water solver in Warp on the staggered raster: uniform rainfall,
  closed BCs, deterministic state-derived Δt, canonical Zarr out (§7.2), live
  float64/Kahan mass balance. **Dam-break validated** (wet-bed Stoker enforced:
  mass 2.5e-9, nRMSE 0.074; dry-bed Ritter diagnostic). The real M0 Smoky Mtns
  tile is steep — LI's worst case — so M1 also has a **mass-conservative per-cell
  flux limiter** (donor-cell β scaling) that keeps depths non-negative out of
  regime; the demo runs stably (mass 2.1e-8, mean depth = rain input). Note: on
  the steep tile mass + spatial pattern are sound, but steep-cell velocities are
  limiter-shaped, **not** validated LI hydraulics — don't carry that as a fidelity
  claim into M2. Runs are bitwise-deterministic (verified). See
  `docs/plans/M1-water-moves.md`.
- **M0 — Foundation: done.** SRTM `N35W083` → conditioned (UTM 17N, sink-fill, D8
  flow dir/accum) → `.r32` tile → static 3D terrain in Godot via Terrain3D
  (Godot 4.7). Tooling locked: **pysheds** (NumPy-2.x `np.in1d` shim in
  `pipeline/_compat.py`) and **Terrain3D**. See `docs/plans/M0-foundation.md`.
- Milestones M0–M7: `docs/plans/roadmap.md`.

## Commands
- `uv sync` — create/refresh the venv (installs deps + `dev` group).
- `uv sync --extra geo` — also installs the offline DEM-conditioning stack
  (rasterio, pyproj, pysheds). **Needed for the pipeline tests** — without it the 5
  `pipeline/test_pipeline.py` tests `importorskip` and silently skip (still "green").
- `uv run python scripts/smoke_test.py` — toolchain self-check (GPU + Zarr).
- `uv run ruff check .` / `uv run ruff format .` — lint / format.
- `uv run pytest` — tests + validation harness (runs on Warp's **CPU** backend, so
  it works in CI without a GPU). Run after `uv sync --extra geo` to exercise the pipeline.
- Pipeline (M0): `uv run python -m pipeline.condition --src <dem> --out <dir>` then
  `uv run python -m pipeline.tile --src <dir> --out data/tiles/demo --single`.
- Solver: `uv run python -m solver.run --config scenarios/demo_basin_rain.toml`
  runs a §7.1 scenario → `results.zarr` + `status.json` + `frames/` (viewer stream)
  + `<store>.provenance.json`. Bare `uv run python -m solver.run` still runs the
  in-code demo; `--tiles`, `--out`, `--status`, `--frames-dir`, `--no-frames`,
  `--end-time`, `--output-every`, `--rain-mm-hr`, `--device` override. Re-export
  viewer tiles from an existing store: `uv run python -m solver.io.viewer_export <zarr> <out_dir>`.
- M3 scenarios: `scenarios/river_reach.toml` (inflow hydrograph + infiltration +
  open boundary, self-contained) and `scenarios/spatial_fields.toml` (Manning +
  infiltration `.r32` fields — generate them first with
  `uv run python scripts/make_demo_fields.py`).
- M4 scenario: `scenarios/river_reach_hllc.toml` — the HLLC fidelity scheme on the
  *same* scenario as `river_reach.toml` (only `[meta].scheme` and the CFL differ), so
  running both gives a like-for-like LI-vs-HLLC side-by-side. Pick the scheme per
  scenario with `[meta] scheme = "local_inertial" | "hllc_fv"`; note HLLC wants a
  tighter `[run] cfl` (~0.45) than LI (~0.7) because its bound is velocity-dependent.
- Viewer (M2): open `viewer/` in Godot 4.7 (main scene `results_view.tscn`) — it
  loads `data/results/frames/` and can launch the solver via the **Run solver**
  button. Headless checks: `godot --headless --path viewer -- --rbverify` (read
  path), `godot --path viewer -- --rbshot[=name.png]` (screenshot),
  `godot --path viewer -- --rblaunch` (full-loop subprocess smoke).

## Conventions
- **Package manager: `uv`** (not pip/venv). Python pinned to **3.13** via
  `.python-version`. Add deps with `uv add`, dev deps with `uv add --dev`.
- **Commits: Conventional Commits** (`feat:`, `fix:`, `docs:`, `chore:`,
  `refactor:`, `test:`). Each commit should pass `ruff` + `pytest`. Scope by
  component where useful (`feat(solver): ...`).
- **Plan before code** for any milestone-sized work; capture the plan in
  `docs/plans/`. Stop at each milestone's demo and confirm before the next
  (HANDOFF §9, §13).
- **Numerics invariants (HANDOFF §8):** float32 GPU fields, but the global
  mass-balance accumulator is **float64 / Kahan**. Adaptive `Δt` derives from
  **state, never wall-clock** (determinism). The mass-balance relative error is a
  hard validation gate — exceedance is a failing test, not a warning.
- **The decoupling contract is sacred (§7):** the solver only ever *writes* files
  (Zarr, per-frame tiles, status.json); the viewer only ever *reads* them. No
  shared memory, no shared process.

## Gotchas (what's easy to get wrong here)
- **`richdem` is dead** — no wheels past cp37, won't build on modern Python. Use
  **pysheds** (or WhiteboxTools) for sink-fill / flow direction / flow accumulation.
- **pysheds 0.5 is stale** (classifiers stop at 3.9). It installs as pure-Python on
  top of numba, but verify it actually *runs* on 3.13 + current numba before
  building M0's pipeline on it; WhiteboxTools is the no-Python-coupling fallback.
- **Windows console encoding:** keep script `print()` output ASCII (no em-dashes) —
  the default code page mangles unicode and clutters logs.
- **Windows atomic-write vs concurrent reader:** `os.replace(tmp, dst)` onto a file
  the viewer holds open for a plain read fails with `PermissionError`/`WinError 5`
  (Godot's `FileAccess` read locks without `FILE_SHARE_DELETE`). This bit
  `status.json` in the M2 loop — retry the replace with a short backoff and treat
  the write as best-effort (`solver/io/status.py`). It's fine on POSIX and invisible
  to standalone runs, so **only the live viewer loop reproduces it** — verify file
  handoffs with `--rblaunch`, not just standalone/`pytest`. Applies to any file the
  viewer reads while the solver writes (frame tiles, `manifest.json`).
- **Well-balancedness is easy to *almost* get, and a fully-wet lake won't catch it.**
  The M4 lake-at-rest keystone passed while MUSCL reconstruction across a **shoreline**
  (a dry neighbour injects a spurious water/bed slope into the minmod stencil) spun a
  smooth bowl up to ~20 m/s. Any change to reconstruction, the bed-slope source, or
  wet/dry handling must stay green on `test_shoreline_lake_at_rest_on_bumpy_bed`, not
  just the flat-lake test — and the first-order drop near dry cells (`hllc._dryfactor`)
  must be applied **identically in the flux and source kernels**, or Audusse's exact
  balance breaks.
- **Never keep depth non-negative with a bare `max(h, 0)`.** It invents mass and
  silently breaks the ledger and any boundary banking (M4 measured ~6.5e-2 on a full
  drain, five orders over the gate). Both schemes use a **donor-cell β mass-flux
  limiter** instead: scale each mass face by its upwind cell's `min(1, h/out_depth)`,
  so the shared face is scaled once by its donor and mass is conserved exactly.
- **Warp JIT cache** lives under `%LOCALAPPDATA%\NVIDIA\warp\Cache`, not the repo;
  first kernel launch pays a ~0.7 s compile, then it's cached.
- **Don't reintroduce real-time** as a primary mode, cross-vendor GPU support, or
  certification-grade accuracy claims (HANDOFF §3, non-goals).
