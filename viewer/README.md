# Viewer (Godot 4.x)

The interactive 3D viewer. It **reads files the pipeline/solver write** — it never
shares memory or a process with the solver (HANDOFF §4, §7.4). Responsibilities:

- Scenario setup UI → writes the scenario TOML.
- Launches the solver as a subprocess (`python -m solver.run --config <path>`).
- Polls `status.json` for progress.
- On completion, streams the per-frame tile manifest and renders the timeline +
  3D camera, with depth/velocity colormaps and a water surface lifted off the bed.

## Scenes

- `scenes/results_view.tscn` — **the main scene** (`scripts/results_player.gd`):
  terrain + lifted water surface + timeline + a *Run solver* button.
- `scenes/terrain_view.tscn` — the M0 standalone terrain check
  (`scripts/terrain_loader.gd`): reads `data/tiles/demo/tiles.json` + the `.r32`
  tile and imports **tile 0** into a Terrain3D node. The tile is raw little-endian
  float32 in **metres**, so it maps 1:1 to a Godot `FORMAT_RF` heightmap image;
  `vertex_spacing = dx` fixes horizontal scale. No height rescaling.

### Where the terrain comes from (results view)

Not from `tiles.json`. From M6 a run's domain is a tile **mosaic**, optionally
windowed and coarsened, so no tile on disk is the surface it stepped on. The solver
therefore exports its own bed into `frames/` (`manifest["static"]`, §7.3) and the
viewer renders **that** — one image, one Terrain3D import, extent/origin/cell size
shared with the water by construction. A results set without a `static` section (an
older export) falls back to M0 tile 0 and warns if it does not cover the run; the
pre-run scene shows tile 0 too, since there is no run yet to define a domain.

## Prerequisites

1. **Godot 4.x** (developed against **4.7-stable**).
2. **Terrain3D** addon — ~50 MB of GDExtension binaries, **not** committed
   (gitignored). Fetch the pinned, 4.7-verified build once:

   ```bash
   bash viewer/scripts/fetch_terrain3d.sh
   ```

   (or install Terrain3D 1.0.2 from the Godot Asset Library into `viewer/addons/`).
3. **Terrain tiles** — produce them with the pipeline:

   ```bash
   uv run python -m pipeline.condition --src data/dem/raw/N35W083.hgt --out data/dem/conditioned
   uv run python -m pipeline.tile      --src data/dem/conditioned --out data/tiles/demo --single
   ```

## Run

```bash
# Open in the editor, then press Play (F5):
godot --path viewer --editor
# …or run the scene directly:
godot --path viewer
```

You should see the conditioned SRTM tile (Great Smoky Mountains, ~29 km across,
365–1564 m) as a lit 3D terrain. **Headless smoke check** (no rendering, just
verifies the import path and that heights are real metres — used by CI):

```bash
godot --headless --path viewer    # prints "headless verify OK (true sampled min=… m)"
```

A windowed one-shot screenshot (writes into `data/tiles/demo/`):

```bash
godot --path viewer -- --rbshot=viewer_screenshot.png
```

## Notes / gotchas

- Terrain3D 1.0.2 keeps `region_size` at 256, so the 1024² tile tiles into a 4×4
  region grid — the seams are continuous (no zero-height cracks; verified by
  sampling). `data.get_height_range().x` reports `0` due to a padding texel; the
  rendered surface min is ~365 m (see the headless `true sampled min`).
- Terrain3D stores its region data under `user://terrain_demo_data` (outside the
  repo), rebuilt on each load from the `.r32` tile. The results view uses
  `user://terrain_results_data` and **clears it before every rebuild**: regions are
  keyed by world position, so importing a smaller domain over a larger one would
  leave the difference standing after a run changes the grid.
- Camera clip planes scale with the domain (`near = dx`, `far = 4 x span`). A fixed
  far plane large enough for a 76.8 km mosaic degenerates the depth range and renders
  *nothing* (Godot's light culler starts refusing frustums) — which `--rbverify`
  cannot see, so check a screenshot when the domain scale changes.
- Godot caches (`.godot/`, `.import/`) and `addons/` are git-ignored. Terrain LOD
  at reach scale is the viewer's one demanding job (HANDOFF §5, §12).
