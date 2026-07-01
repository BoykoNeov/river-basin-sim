# M2 — The Loop Closes

**Goal:** turn the M1 solver into a real **configure → run → explore** sandbox by
implementing the §7 decoupling contracts end to end and standing up the Godot
viewer that reads them. After M2 a user edits a TOML, the viewer launches the
solver as a subprocess, watches progress via `status.json`, and on completion
scrubs a timeline of the flood over the real M0 terrain with a depth colormap and
a lifted 3D water surface.

Depends on: M1 (local-inertial solver, canonical Zarr, mass balance — done). Gate
before M3.

---

## 0. Scope — what M2 is and is *not*

**In (HANDOFF §9 M2, roadmap):**
- §7.1 **config-in**: TOML scenario loader → the solver's `Scenario`.
- §7.4 **subprocess + `status.json`**: `python -m solver.run --config <toml>`,
  periodic status writes, non-blocking launch + poll from Godot.
- §7.3 **per-frame viewer tiles**: raw little-endian float32 + `manifest.json`,
  produced as a **post-process pass over the canonical Zarr**.
- **Godot viewer**: timeline scrubber, depth colormap, lifted water surface over
  the M0 terrain. The read-only half of the contract (§4).

**Explicitly deferred (do not build now):**
- **Spatially-varying parameter fields, inflow hydrographs, open/fixed-stage BCs,
  structures, command log → M3.** M2 parses the *full* §7.1 schema but **supports
  only** the M2 subset (uniform rain, closed BCs, scalar `manning_n`). Any M3/M4
  field present in a config is a **loud error**, not a silent ignore — that error
  *is* the scope gate.
- **HLLC FV (`scheme = "hllc_fv"`) → M4.** Reject it in the loader.
- **Multi-tile splitting / tiling-at-scale → M6.** The manifest is tile-*aware*
  (a 1×1 grid for the demo tile) but we do not build the splitter.
- **Scenario-setup UI in Godot → M3.** M2's viewer launches an existing TOML and
  explores results; it does not author scenarios with a GUI.

**Files built this milestone (paths per HANDOFF §6):**
```
solver/io/config.py            # §7.1 TOML -> Scenario (+ validation / scope gate)
solver/io/status.py            # §7.4 status.json writer (states + progress)
solver/io/viewer_export.py     # §7.3 Zarr -> per-frame raw tiles + manifest.json
solver/run.py                  # gain --config; write status.json; call viewer_export
solver/io/test_config.py       # loader + scope-gate tests
solver/io/test_viewer_export.py# byte-layout + manifest tests
solver/io/test_status.py       # status.json shape/state-sequence test
scenarios/demo_basin_rain.toml # the canonical example config (§7.1)
viewer/scripts/run_controller.gd   # subprocess launch + status.json polling
viewer/scripts/results_player.gd   # frame streaming + timeline + water mesh
viewer/shaders/water_surface.gdshader  # depth colormap + lifted surface + dry mask
viewer/scenes/results_view.tscn        # the explore scene (terrain + water + UI)
```

---

## 1. The contracts (HANDOFF §7) — exact formats

### 1.1 Config (§7.1) — `solver/io/config.py`
Parse the full schema; **support** the M2 subset. Map into the existing `Scenario`
dataclass (extended as needed). Enforced this milestone:

| TOML | M2 handling |
|---|---|
| `[meta] name, seed, scheme` | `scheme` must be `"local_inertial"`; else error. `seed` recorded (unused — solver is already deterministic). |
| `[grid] tiles_dir, dx, crs` | `dx`/`crs` default from the tile `tiles.json` manifest; TOML overrides. |
| `[run] end_time, output_every, cfl, dt_max` | `cfl` maps to the LI `alpha`. |
| `[rainfall] type, rate_mm_hr, duration_s` | `type` must be `"uniform"`; else error (M3). |
| `[parameters] manning_n` | **scalar only**; a path/string value → error (M3 fields). `infiltration` present → error. |
| `[boundaries] default` | must be `"closed"`; anything else → error (M3). |
| `[[structures]]` | present → error (M5). |

Validation is explicit and message-bearing (name the field + the milestone that
adds it). Unknown top-level keys warn; unknown keys in a supported table warn.

### 1.2 status.json (§7.4) — `solver/io/status.py`
`{state, progress, sim_time, eta_s, message}`, `state ∈ {starting, running,
writing, done, error}`. Written atomically (temp file + replace) so the viewer
never reads a half-written file. `progress ∈ [0,1] = sim_time / end_time`. `eta_s`
from **wall-clock** — permitted here because it never feeds Δt or the Zarr (stays
the honest side of the determinism line, §12). On exception: `state="error"`,
`message` = the exception string; re-raise after writing.

Cadence: `starting` at launch, `running` at each output frame, `writing` before
the viewer-export pass, `done` at the end, `error` on failure.

### 1.3 Per-frame viewer tiles (§7.3) — `solver/io/viewer_export.py`
A **post-process over the finished Zarr** (own function + CLI `python -m
solver.io.viewer_export <zarr> <out_dir>`), also called at the end of a run. For
the single demo tile, one `.raw` per frame per exported field.

```
frames/
├── manifest.json
├── f0000_depth.raw   # raw little-endian float32, row-major (Y,X), one tile
├── f0001_depth.raw
└── ...
```

`manifest.json`:
```json
{
  "dx": 28.146, "crs": "EPSG:32617",
  "grid": {"width": 1024, "height": 1024},
  "tile_grid": {"cols": 1, "rows": 1},
  "fields": ["depth"],
  "h_dry": 1e-3,
  "global": {"depth": {"min": 0.0, "max": 8.9, "p50": 0.001, "p99": 0.5}},
  "frames": [
    {"index": 0, "time": 0.0,
     "files": {"depth": "f0000_depth.raw"},
     "depth": {"min": 0.0, "max": 0.0}},
    ...
  ]
}
```

- **depth only** for M2 (η reconstructed in-shader from the static bed the viewer
  already loads; u/v deferred to when the viewer draws velocity).
- **Global** robust stats (min/max + p50/p99 over all wet cells across all frames)
  drive a *stable* colormap; per-frame min/max also carried. Colormap default uses
  a clamped range (0 → global p99) so the thin floodplain sheet is visible and the
  rare deep channel doesn't wash it out. The steep M0 tile's max ~8.9 m vs p99
  ~0.5 m is exactly why per-frame-max coloring is wrong.
- Bytes match the M0 `.r32` convention (raw LE f32, row-major, metres) so Godot
  loads them into `FORMAT_RF` images with the **same orientation/origin** the M0
  terrain loader established — no transpose.

---

## 2. Godot viewer — explore scene

Reuse the M0 terrain load path (Terrain3D, `vertex_spacing = dx`, `FORMAT_RF`
import) for the static bed. Add on top:

### 2.1 `run_controller.gd` — launch + poll (§7.4)
- Write/point at the TOML; launch `python -m solver.run --config <toml>` with
  **`OS.create_process`** (non-blocking) using the `uv` venv interpreter; store the
  PID. Windows cwd + interpreter path is the integration-friction point — resolve
  the repo-root venv `Scripts/python.exe`, fall back to `uv run`.
- Poll `status.json` on a `Timer` (~4 Hz): show `state`/`progress`/`eta_s` in a
  small HUD. On `done`, hand `frames/manifest.json` to the player and enable the
  timeline. On `error`, surface `message`.

### 2.2 `results_player.gd` — timeline + streaming
- Load `manifest.json`; build an `HSlider` (0 … n_frames−1) + play/pause.
- On frame change, read that frame's `depth` `.raw` into a `FORMAT_RF` `ImageTexture`
  and set it on the water shader. Small tile (1024²·4 B = 4 MB/frame) → load on
  demand; cache a few. No per-frame data scan — colormap ranges come from the
  manifest.

### 2.3 `water_surface.gdshader` — colormap + lifted surface + dry mask
- A subdivided `PlaneMesh` (or Terrain3D-aligned grid) covering the domain, same
  `dx`/origin as the terrain.
- **Vertex:** sample depth texture; lift each vertex to `η = bed + depth`. Bed from
  the same heightmap the terrain uses (a second `FORMAT_RF` texture) so water and
  terrain register exactly.
- **Fragment:** `discard` where `depth < h_dry` (matches the solver's dry cell) so
  dry ground shows terrain, not a zero-depth sheet. Color by depth via a colormap
  (viridis-like) over the clamped `[0, global.p99]` range from the manifest; slight
  transparency + fresnel for a water look.

---

## 3. Build order (each step keeps `ruff` + `pytest` green)

1. **`config.py` + `test_config.py`** — loader, `Scenario` extension, scope-gate
   errors. `scenarios/demo_basin_rain.toml` reproduces the current in-code demo.
2. **`status.py` + `test_status.py`** — atomic writer, state sequence.
3. **`viewer_export.py` + `test_viewer_export.py`** — Zarr → tiles + manifest;
   byte-layout + robust-stats + orientation tests.
4. **Wire `run.py`** — `--config`; write `status.json` through the run; call
   `viewer_export` in the `writing` phase. Regenerate the demo `results.zarr` +
   `frames/`. **Confirm checkpoint:** show the manifest + a rendered depth PNG
   (matplotlib) before touching Godot — proves the bytes are right.
5. **Godot `results_view.tscn` + scripts + shader** — build against the real
   `frames/` from step 4. Headless verify (load manifest, load a frame, assert the
   water texture is non-empty) + a windowed screenshot for visual proof.
6. **Full-loop smoke** — from Godot: launch the solver on the demo TOML, poll to
   `done`, auto-load results, scrub. Screenshot the flooded valley.

---

## 4. Acceptance / demo — MET (2026-07-01)

- [x] `python -m solver.run --config scenarios/demo_basin_rain.toml` runs on the
      GPU (RTX 5090), writes `results.zarr`, `status.json` (walking
      `starting → running → writing → done`), and `frames/` + `manifest.json`
      (13 frames).
- [x] Config scope gate: an M3/M4 field (open BC / parameter raster / `hllc_fv` /
      structure) fails with a clear, milestone-naming `ConfigError` — verified both
      as a unit test and at runtime (the error is written to `status.json` as
      `state="error"`, not a silent exit, so the viewer never hangs).
- [x] `viewer_export` bytes round-trip: every frame `.raw` reloaded equals the
      Zarr `depth[i]` (shape/orientation/values); manifest global max exact, p50 ≤
      p99 ≤ max. Demo tile: global p99 = 0.83 m vs max 10.49 m → the clamped
      `[0, p99]` colormap keeps the floodplain visible.
- [x] Mass-balance gate still `< 1e-6` on the config-driven run (**2.12e-8**;
      physics unchanged — status/export are read-only observers).
- [x] Godot: **launches the solver subprocess** (Windows batch + `uv run`), polls
      `status.json` at 4 Hz, auto-loads on `done`; timeline scrubs 13 frames; depth
      colormap + lifted water surface **register with the terrain** (water sits
      conformally in the valleys), dry ground shows terrain. Windowed screenshot
      captured (`data/results/results_screenshot.png`).
- [x] `ruff` + `ruff format` clean; `pytest` green (**38 tests**, new
      config/status/viewer_export/error-status tests included). Headless viewer
      verify green (`--rbverify`).
- **Stop and confirm before M3.** ← we are here.

### Orientation is verified, not assumed
The water registers with the terrain *by construction*: the shader samples
`bed_tex` and `depth_tex` at the **same UV**, and the round-trip test proves the
depth `.raw` shares the bed's row-major layout and dimensions. The only free
variable — PlaneMesh UV vs ImageTexture UV convention — also rides on `bed`, so
any flip/transpose would compute `η = bed + depth` from a *mismatched* bed cell
and float water over ridges or bury it in the wrong cells. The screenshot shows
water sitting conformally in the valleys, which is only possible if the UV
convention is correct → `flip_v = false` is confirmed.

---

## 5. Notes / risks
- **Viewer shader is the risk sink (HANDOFF §12).** Mitigated by building Godot
  against real on-disk bytes (step 4 before step 5), reusing the M0 orientation
  convention, and reconstructing η in-shader from the shared bed texture.
- **Windows subprocess launch** (uv venv interpreter, cwd) is the integration
  friction — resolve the interpreter explicitly, keep a `uv run` fallback.
- **Determinism** unaffected: `status.json`/export are read-only observers of a
  run whose Δt still derives from state. `eta_s` wall-clock stays out of the sim.
</content>
</invoke>
