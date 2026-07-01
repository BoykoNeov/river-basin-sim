# M0 — Foundation

**Goal:** prove the pipeline + viewer + file-handoff end to end. Load a sample DEM,
condition it (sink-fill + D8 flow direction + flow accumulation), tile it, and load
one terrain tile as a static 3D heightmap in Godot. **No dynamics yet.**

## Done (initialization batch)

- [x] Repo scaffold per HANDOFF §6 (package skeleton, docs, `.gitignore`).
- [x] `uv` environment, Python pinned to **3.13**; deps: warp-lang 1.14, numpy,
      zarr, xarray. dev group: pytest, ruff.
- [x] Toolchain smoke test (`scripts/smoke_test.py`): Warp initialises, sees the
      RTX 5090, JIT-compiles an **sm_120** kernel, Zarr↔xarray round-trip. **Green.**
- [x] Public GitHub repo created; CLAUDE.md + README + roadmap committed.

## Next (M0-pipeline batch) — DONE (2026-06-22)

### 0. Tooling decision: pysheds vs WhiteboxTools — **pysheds** (chosen)
- [x] Verified `pysheds` 0.5 runs on Python 3.13 / numba 0.65.1 / NumPy 2.4.6. One
      catch: it still calls the **removed `np.in1d`** (NumPy 2.0) in 9 `sgrid.py`
      sites. `np.isin` is an exact drop-in (all sites pass `fdir.ravel()`), applied
      as a one-line shim in `pipeline/_compat.py` (imported at package init).
      WhiteboxTools remains the documented fallback. Recorded in `pipeline/sources.md`.

### 1. DEM conditioning (`pipeline/condition.py`)
- [x] Sample DEM: **SRTMGL1 `N35W083`** (Great Smoky Mtns, public domain, ESA STEP
      mirror, no login). Provenance in `pipeline/sources.md`. `data/` is gitignored.
- [x] Reproject to **UTM 17N (EPSG:32617)**, dx≈28.15 m → sink-fill → resolve-flats
      → D8 flow dir → flow accumulation. Writes `bed_elevation`, `filled_elevation`,
      `flow_direction`, `flow_accumulation` GeoTIFFs + `condition.json`.
- [x] Design note: **bed = true DEM** (physical bed for the solver/viewer); the
      sink-filled surface is kept separate and used *only* to derive the flow
      network. Validated: max accumulation 1.72M cells → one coherent dendritic net.

### 2. Tiling (`pipeline/tile.py`)
- [x] Cuts the conditioned bed into **raw little-endian float32 `.r32` tiles**
      (HANDOFF §7 byte layout) + a `tiles.json` manifest (dx, CRS, bounds, h_min/max).
- [x] One artifact serves both consumers: the solver memory-maps the `.r32` as its
      bed `z`; the viewer builds a Godot `FORMAT_RF` heightmap from the same bytes.
- [x] `--single` auto-picks a clean interior tile on the strongest river (demo:
      1024², 365–1564 m, zero nodata).

### 3. Godot terrain scene (`viewer/`) — **Terrain3D** (chosen)
- [x] Terrain decision: **Terrain3D** plugin (per the developer). Verified it loads
      and instantiates on **Godot 4.7** (compat floor 4.4; empirically clean).
- [x] `viewer/` Godot 4 project loads the `.r32` tile into a Terrain3D node as a
      static heightmap. `vertex_spacing = dx` + true metre heights → correct relief.
      Headless smoke check + windowed screenshot both confirm. Addon is gitignored
      (~50 MB); `viewer/scripts/fetch_terrain3d.sh` re-fetches the pinned build.

## Acceptance / demo — MET
- [x] One sample DEM → conditioned → tiled, with provenance recorded.
- [x] The conditioned tile opens as a 3D terrain in Godot (screenshot captured).
- [x] Smoke test green; `ruff` (+ `ruff format`) + `pytest` clean (5 pipeline tests).
- **Stop and confirm before M1** (local-inertial solver). ← we are here.

## Open questions for the developer
- ~~License~~ → **BNCL-1.0** (Boyko Non-Commercial License v1.0; replaced the
  initial MIT choice). Copyright "Boyko Neov".
- ~~Sample DEM~~ → SRTM `N35W083` (Great Smoky Mtns), chosen 2026-06-22.
- ~~Terrain rendering~~ → **Terrain3D** (chosen 2026-06-22); verified on Godot 4.7.
