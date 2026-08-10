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
- **M6 — Reach: done (signed off on GPU + Godot, 2026-08-09).** Reach is bought by **choosing the
  resolution** and putting the lost river back, not by nesting grids. Three pieces:
  **`solver/io/mosaic.py`** makes the domain the whole **tile mosaic** (`[grid] tiles`
  = `all`/`first`, `[grid] window` = an inclusive box in mosaic coords; only the tiles a
  window touches are memmapped; uncovered cells are filled flat **and counted**);
  **`solver/io/coarsen.py`** runs `k`× coarser (`[grid] coarsen`, `dx' = k·dx`) with every
  field aggregated **once, before any water moves** — block **mean** for the bed and the
  rate fields (volume-preserving), block **max** for channel width/depth (a river crosses
  a block; a mean thins it away) — and cell indices mapped `i // k`, refused if they leave
  the domain; and **`solver/core/channels.py`** carries a **channel narrower than its
  cell**. That last is the milestone: `h` keeps its meaning (volume per unit plan area,
  so continuity, the donor limiter and the ledger are untouched and **mass conservation
  stays exact by construction**), and the channel enters only through the **storage
  curve** `h → η` (below bank full `η = z − d + h·dx/w`; above it `η = z + (h − h_bf)`;
  `w = 0` collapses to `z + h` bit for bit) and a **two-component face update** — channel
  flow with hydraulic radius `A/P` **not** the depth, plus the M1 floodplain flow,
  recombined into the same total flux. The CFL reduces over the water **column**, since a
  channel concentrates depth by `dx/w`. **Local-inertial only, loudly** (the mirror of
  M5's HLLC-only `fixed_stage`). **Validated:** normal depth in a 20 m channel inside 50 m
  cells is **−0.1%** off its own analytical section (1.296 vs 1.298 m); overbank spill
  conserves mass (3.6e-8); and the **reach claim** — the same 2 km reach resolved at 10 m
  (2000 cells) versus sub-grid at 100 m (20 cells) — agrees to **0.1%** in depth. §7.3's
  `tile_grid` is real now (frames > 512 split row-major; an untiled export is byte-identical
  to M2's), and the Godot reader takes its grid from the manifest and blits tiles —
  **now verified on hardware**: all 25 frames reassemble byte-exactly to the Zarr,
  `--rbverify` reads the coarsened mosaic, `--rbshot` renders `h_max` to the pixel, and
  `--rblaunch` drives the full subprocess loop (2.12e-8 at sign-off; **2.59e-8** since the
  precision pass, which by design ends the bit-for-bit identity with M2's figure), with
  the Windows `os.replace` race not firing across 4× the file handoffs. Demo
  `scenarios/reach_basin.toml`: a **6×6 mosaic, 76.8 km square**, run at 768² @ 100 m with
  2232 channel cells, mass **7.21e-8** on CPU / **1.60e-7** on the 5090 (was 2.79e-7 /
  3.08e-7 before the precision pass). **234 tests green.** M6's loud carried finding — the
  same demo at `coarsen = 4` exceeding the gate — **is fixed**: see the precision pass
  below. See `docs/plans/M6-reach.md`.
- **Precision pass — compensated areal sources: done (2026-08-09).** The first of M6's two
  carried debts, landed before M7 puts a second distributed source on the same fields.
  float32 `h += rate*dt`, once per cell per step, was what put a reach-scale rain-on-grid
  run at the mass gate for *arithmetic* reasons — measured, not guessed: the residual
  climbed ~220 m³ per 1800 s **while it rained**, hit 647 m³ when the storm stopped, then
  went flat over the next 10 h with water still moving and draining. Fluxes were never the
  leak. **`solver/core/sources.py`** gives each cell its own float32 **Kahan compensation**
  term — the ledger's own idiom (`massbalance._Kahan`) moved onto the grid — so the bits an
  add discards are repaid by the next one. `h` stays float32 (**§2 untouched**, no field
  promoted, one extra f32 array). Armed **only when rain actually falls**, so every run
  without an areal source is **bitwise unchanged** (dam-break, lake-at-rest, the EA
  benchmarks, M5's `reservoir_release`); **point sources are deliberately out of scope**
  (inflow measured ~1.3% of the residual, and arming on it would perturb rain-free
  scenarios). **Result: 3.77e-6 → 1.28e-7** on the failing `coarsen = 4` case — 647 m³ of
  storm drift becomes −3.15 m³, and the ~21 m³ left is flux/limiter round-off, the floor a
  source-only fix can reach. Every rain-bearing figure was re-measured (all improved except
  the M1/M2 demos, 2.12e-8 → 2.59e-8 — at that magnitude source drift was never what set
  them). Tests gate **ratios, not thresholds**, plus a **fast-math canary** asserting the
  compensation term is nonzero, because reassociating it away would make every other
  assertion measure an uncompensated add against itself. See `docs/plans/precision-sources.md`.
- **Viewer terrain mosaic: done (2026-08-09).** M6's second carried debt. The viewer's
  terrain was still M2's tile-0 load, so a mosaic run's (correct) water rendered over
  one patch of a possibly unrelated DEM. The fix is a contract addition, not a GDScript
  mosaic assembler: **`solver/io/viewer_export.py`** ships the canonical store's `bed`
  into `frames/` as **`manifest["static"]`**, through the *same* tile layout and the
  same entry shape as a frame — the bed is already mosaic-assembled, windowed,
  coarsened, gap-filled and datum-un-shifted, so terrain and water share extent, origin
  and cell size **by construction** (§7.3 records this; `manifest["domain"]` carries the
  assembly record so **gap fill renders as declared, not as a mystery plateau**).
  **`results_player.gd`** builds terrain *from results* and is **re-entrant** — a
  finishing run can change the grid under a live viewer — with one `_apply_geometry`
  fitting terrain, water plane, `bed_tex` (the shader lifts `η = bed + depth`, so the
  wrong bed was mis-lifting the surface, not just the backdrop) and camera together.
  The M0 tile stays as the pre-run/legacy fallback and **warns** when it does not cover
  the run. **Validated:** `--rbverify` now gates registration — same grid, same `dx`,
  and the *imported* surface sampled and bracketed against the exported bed
  (768x768 @ 100.00m, sampled 86.4..260.0 m vs bed 85.0..260.0 m) — where the old
  check passed on the broken composite; note `get_height_range().x` reads 0 off a
  padding texel, so it cannot be used for this (its "relief" is the max elevation);
  `reach_basin` re-run on the 5090 at mass **1.60e-07** (unchanged) renders the whole
  76.8 km basin; the M2 demo is unchanged and `--rblaunch` reaches `done` at 2.59e-08.
  **241 tests green.** Carried: the shader still lifts a sub-grid channel by
  `bed + depth`, not its storage curve, so the sheet sits up to `d` high on the river.
  See `docs/plans/viewer-terrain-mosaic.md`.
- **M5 — Multi-physics: done.** The locked time-integration
  decision (HANDOFF §2/§8) is now real code. **`solver/scheduler.py`** owns the single
  simulated clock and *only* that: the fast scheme still computes its own state-derived
  `Δt`; the scheduler clamps it so a step never crosses a **sync point** (output cadence,
  forcing breakpoints, `end_time`, **slow-process activations**) and yields a `Tick`.
  It is a **clock, not a driver** — `run.py` keeps state/stepping/forcing/accounting/IO,
  which is what makes the sync-point algebra unit-testable with a stub `dt_fn` and no GPU.
  With no slow processes the event set and its arithmetic are unchanged, so **pre-M5 runs
  are bitwise-identical** (verified against stored LI/M3/HLLC baselines; guarded in-tree
  by a test that replays the pre-M5 inline loop as an executable reference).
  Exercised by **reservoir operations** (`[[structures]]` + `solver/processes/reservoir.py`):
  a structure is **barrier geometry plus a rule** — cells' bed raised to `crest_m` (so
  impoundment and overtopping are ordinary solver physics, nothing to re-validate) plus
  an optional release rule (`fixed` open-loop, `target_stage` proportional draw-down)
  evaluated **only at slow-clock activations** while the flood scheme sub-cycles freely.
  The pool→outlet transfer is **mass-exact** (withdrawal banks the actual f32 depth
  change in f64, the request is capped by what the pool holds, delivery banks its
  rounding); the outlet is a **box**, because splitting hands over a whole interval at
  once and 54,000 m³ into one 40 m cell is a 34 m column. Also lands the two M4
  deferrals: **`fixed_stage`** — a prescribed-surface Dirichlet ghost, the third member
  of M4's per-edge ghost family, constant or piecewise-linear in time, banked in *both*
  directions, **HLLC-only and loudly so** (LI has no boundary face to prescribe a surface
  on) — and the **datum shift** (`[grid] datum`, `z' = z − z_ref`), which protects float32
  `η = h + z` at altitude and un-shifts the bed on the way out. **Validated:** stage edge
  leaves a lake at rest (1.2e-5 m/s, nothing crosses), per-edge fill ×4 settles exactly at
  the prescribed level, a stage drain to 90% dry cells still banks exactly (7.3e-8);
  release transfer moves 300.000 m³ with a 0.0 residual, `target_stage` converges
  3.76→3.002 m with Q easing 3.04→0.01 m³/s, activations are bit-identical at `dt_max`
  5.0 and 1.0; and **EA SC080035 Test 1** (disconnected water body) — the far pond floods
  to 0.393 m and still holds 0.317 m after 5 h at the 10.10 m ridge crest while the
  connected pond returns exactly to the 9.700 m boundary. Demo
  `scenarios/reservoir_release.toml` fills a valley reservoir to 0.5 Mm³ then draws it
  down under proportional control (Q easing 40.8 → 12.7 m³/s as the stage falls toward
  its target). **Signed off on GPU + CPU, 2026-08-09, out of order — after M6**: mass
  **1.36e-7 CPU / 3.15e-7 CUDA** (a 1.8e-7 backend delta, so reduction order is not what
  sets it), pool peaks at 77.04 m under the 78 m crest and never overtops, the rule
  engages at 75.11 m and eases monotonically to 12.7 m³/s while the stage falls to
  75.63 m. Because M6 refactored `solver/io/` underneath this scenario, that run is also
  a **regression check on the mosaic loader**. **189 tests green** at authoring, 228 at
  sign-off.
  **Two carried honesty notes:** EA Test 1's SC080035 figures were not reachable from
  this session, so its geometry is *faithful in form, reconstructed in detail* (the
  docstring separates pinned from reconstructed; the gate is the qualitative finding
  plus the mass gate); and M4's expectation that Test 1 *needs* the datum shift **does
  not hold** — parametrized over both datums, the runs agree to three decimals and both
  clear the gate (the shift earns its keep at ~500 m+, measured ~1600× at 5000 m).
  See `docs/plans/M5-multi-physics.md`.
- **M4 — Fidelity step: done.** A second, higher-fidelity
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
  6.66e-7 vs the LI baseline's 1.24e-7 on the same scenario (**1.31e-7 vs 1.68e-8**
  since the precision pass — both scenarios carry rain). No viewer change — the
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
  `spatial_fields` 7.57e-8; **1.68e-8 / 7.36e-9** since the precision pass). 82 tests green. See `docs/plans/M3-real-scenarios.md`.
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
  on the RTX 5090; mass gate 2.12e-8 (**2.59e-8** since the precision pass — the loop
  itself is unchanged and still green); error path writes `state="error"` (viewer
  never hangs); 38 tests green. See `docs/plans/M2-loop-closes.md`.
- **M1 — Water moves: done.** Local-inertial (Bates
  2010) shallow-water solver in Warp on the staggered raster: uniform rainfall,
  closed BCs, deterministic state-derived Δt, canonical Zarr out (§7.2), live
  float64/Kahan mass balance. **Dam-break validated** (wet-bed Stoker enforced:
  mass 2.5e-9, nRMSE 0.074; dry-bed Ritter diagnostic). The real M0 Smoky Mtns
  tile is steep — LI's worst case — so M1 also has a **mass-conservative per-cell
  flux limiter** (donor-cell β scaling) that keeps depths non-negative out of
  regime; the demo runs stably (mass 2.1e-8, **2.59e-8** since the precision pass;
  mean depth = rain input). Note: on
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
- M6 scenario: `scenarios/reach_basin.toml` — a 6x6 tile mosaic (1536^2 @ 50 m = 76.8 km
  square) run at `coarsen = 2` (768^2 @ 100 m) with the river carried by sub-grid
  channels. Generate its synthetic basin + channel fields first with
  `uv run python scripts/make_reach_demo.py`. For a real DEM, derive the channel fields
  from the M0 flow accumulation: `uv run python -m pipeline.channels --src <conditioned>
  --tiles <tiles> --out <fields>` (its hydraulic-geometry coefficients are **regional
  calibration inputs**, recorded beside the fields).
- M5 scenario: `scenarios/reservoir_release.toml` — a dam with a `target_stage` release
  rule on the slow clock, a tidal `fixed_stage` southern edge, and inflow hydrographs
  filling the reservoir. Generate its synthetic valley tile first with
  `uv run python scripts/make_reservoir_demo.py` (the dam is placed by cell index and
  crest elevation, so it needs terrain whose valley is known, not arbitrary DEM).
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
- **The mass gate is a *conditioning* gauge as well as a correctness one.** A
  centimetre-thin sheet on a bed at ~1 m loses enough of `h` inside float32
  `η = h + z` that a few hundred steps of accumulated round-off drifts a *closed*
  domain past the 1e-6 gate — with nothing wrong in the physics. Before treating a
  gate failure as a bug, check the depth-to-elevation ratio and step count; the fix is
  usually `[grid] datum` or a better-scaled test, not a change to the scheme. (M4's EA
  Test 2 note is the same effect measured on horizon length.)
- **A clamped step is not a free step, and no gate can see the damage.** The
  scheduler clamps every step to land on a sync point (`dt = min(dt, next_sync - t)`
  — output cadence, forcing breakpoints, slow-process activations; M1–M4 did the same
  inline). That hands local-inertial an abrupt shorten-then-restore, which excites a
  short-wavelength mode. Measured on a uniform steady reach with **no sediment armed
  at all**: interior depth ripple 0.010 mm unclamped, 14 mm at a 45 s cadence, 74 mm
  at 22.5 s, **2342 mm at 11.25 s** — while the mass balance reads 1e-8 throughout,
  because mass *is* conserved and the water is merely in the wrong places. Isolated
  to the scheme by two controls (a closed box with no inflow and no open edge does it
  too; smooth drift of the state-derived `dt` does not). Existing scenarios sit at a
  900 s cadence (0.165 mm), which is why nothing caught it. **So: a frequent sync
  cadence is not the safe direction** — before shortening `output_every` or a slow
  process's interval, check a quantity that a ripple would break, not the mass gate.
  A measured candidate fix (fill each span with `ceil(span/dt)` *equal* steps) is
  recorded but unshipped, because it moves every run's `dt` sequence and with it the
  pre-M5 bitwise-identity invariant. Full tables + controls: `docs/plans/M7-morphology.md` §4.
- **A slow process hands over a whole interval at once — that is the splitting, not a
  bug, but it has a scale.** 60 m³/s over a 900 s reservoir interval is 54,000 m³; into
  a single 40 m cell that is a 34 m instantaneous column. Deliver over a *reach* (the
  `outlet` box), and sanity-check `Q · interval_s / (area · cells)` before believing a
  release-driven result.
- **A distributed source is an accumulator, and float32 accumulators need
  compensating.** A rain-on-grid run adds `rate*dt` to every cell, every step: at reach
  scale that used to drift a *closed* domain past the gate with nothing wrong in the
  physics (`reach_basin` at `coarsen = 4`, 3.77e-6, identically with channels off).
  **Fixed** by per-cell Kahan compensation on the areal sources
  (`solver/core/sources.py`, 1.28e-7), so the envelope is now wide — **but the lesson
  generalizes and M7's sediment is the next case**: any new distributed source added to
  a field must go through `sources.py`, not a bare `+=`, or it reintroduces exactly this.
  Two things the fix does *not* cover: **flux-divergence** round-off (untouched, and now
  the floor — ~21 m³ on that run) and **point sources** (inflow is deliberately
  uncompensated, so rain-free scenarios stay bitwise). Diagnosis discipline still
  applies: check storm depth, cell count and step count before suspecting the scheme.
  See `docs/plans/precision-sources.md`.
- **Every scenario writes to the same default output, and the frames export never
  purges it.** `[meta] name` does not pick the output path — `reach_basin` and
  `demo_basin_rain` both land in `data/results/demo.zarr` + `frames/`. Run them back to
  back and the directory holds two grids at once: after the M6 sign-off, 61 of 113
  `.raw` files were orphans, including 48 **768²-geometry tiles beside a 1024²
  manifest**, same naming, told apart only by byte size. `manifest.json` is the only
  record of which files belong to the current run — never reconstruct frame filenames
  by index, never read a directory listing as the last run's output, and pass `--out` /
  `--frames-dir` when you care about keeping a result. The bed tiles are the sharpest
  case: `bed.raw` / `bed_r00_c00.raw` carry no frame index, so an untiled run's
  `bed.raw` sits beside a tiled run's `bed_r00_c00.raw` and only the manifest says
  which is live (the reader's byte-size check is the backstop, and it yields a hole,
  not an error).
- **The viewer's terrain is the run's bed, and only what the manifest ships registers.**
  It used to be tile 0 of `data/tiles/demo`, which put a 76.8 km mosaic's water over a
  28.8 km patch of an unrelated DEM — both halves right, composite broken. Now
  `viewer_export` ships the store's `bed` as `manifest["static"]` and the viewer renders
  *that* (`_apply_geometry` is the single place terrain, water, `bed_tex` and camera are
  fitted). Consequences: a store exported before this falls back to tile 0 and **warns**
  — re-export rather than trusting the picture; anything new the composite depends on
  must travel through the manifest the same way. Verified-water and aligned-with-terrain
  are still separate claims — `--rbverify` now asserts the second one, and it took a
  *screenshot* to catch an empty render that the headless check called OK.
- **A sub-grid channel conveys only where it is continuous.** `w_face = min(w_L, w_R)`, so
  a channel band that steps sideways faster than it is wide has a wall across it and
  nothing in the depth field says so. Check connectivity when authoring channel geometry
  (`scripts/make_reach_demo.py::check_continuity` exists because the first demo did this).
- **Never keep depth non-negative with a bare `max(h, 0)`.** It invents mass and
  silently breaks the ledger and any boundary banking (M4 measured ~6.5e-2 on a full
  drain, five orders over the gate). Both schemes use a **donor-cell β mass-flux
  limiter** instead: scale each mass face by its upwind cell's `min(1, h/out_depth)`,
  so the shared face is scaled once by its donor and mass is conserved exactly.
- **Warp JIT cache** lives under `%LOCALAPPDATA%\NVIDIA\warp\Cache`, not the repo;
  first kernel launch pays a ~0.7 s compile, then it's cached.
- **Don't reintroduce real-time** as a primary mode, cross-vendor GPU support, or
  certification-grade accuracy claims (HANDOFF §3, non-goals).
