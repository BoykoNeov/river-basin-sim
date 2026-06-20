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

## Next (M0-pipeline batch)

### 1. DEM conditioning (`pipeline/condition.py`)
- [ ] **Decision first:** verify `pysheds` 0.5 actually runs on Python 3.13 + current
      numba with a tiny DEM. If it fights us, switch to **WhiteboxTools** (standalone
      binary, no Python-version coupling). Record the choice in `pipeline/sources.md`.
- [ ] Pick a small sample DEM (SRTM/3DEP/Copernicus tile — small, permissive license).
- [ ] Reproject to a metric CRS; sink-fill; D8 flow direction; flow accumulation.

### 2. Tiling (`pipeline/tile.py`)
- [ ] Cut the conditioned raster into engine-ready tiles (record `dx`, CRS, bounds).
- [ ] Export bed elevation in a form the solver's `grid`/`state` will consume and
      that the viewer can load as a heightmap.

### 3. Godot terrain scene (`viewer/`)
- [ ] Minimal Godot 4.x project: load one terrain tile as a static 3D heightmap.
- [ ] Decide terrain approach now or defer: `Terrain3D` plugin vs custom clipmap
      (the viewer's one demanding job at reach scale — HANDOFF §12).

## Acceptance / demo
- One sample DEM → conditioned → tiled, with provenance recorded.
- The conditioned tile opens as a 3D terrain in Godot.
- Smoke test still green; `ruff` + `pytest` clean.
- **Stop and confirm before M1** (local-inertial solver).

## Open questions for the developer
- **License** for the public repo (currently none → all-rights-reserved). MIT /
  Apache-2.0 / other?
- Sample DEM region/source preference, or pick a small public-domain SRTM tile?
- Terrain rendering: `Terrain3D` plugin vs hand-rolled clipmap for M0 (a flat
  heightmap is enough for the M0 demo; LOD can wait).
