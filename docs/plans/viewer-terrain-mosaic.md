# Viewer terrain mosaic — the terrain follows the run

**Status: done, 2026-08-09.** The second of M6's two carried debts (`roadmap.md` →
"Carried debts before M7"), landing after the precision pass. Not a milestone: a
scoped fix to one seam — which surface the viewer renders under the water.

---

## 1. The defect

M6 made the run domain the **tile mosaic**: `scenarios/reach_basin.toml` steps a
768² grid at 100 m covering 76.8 km, assembled from 36 tiles and coarsened by 2. The
water sheet followed — `results_player.gd` took its geometry from the frame manifest
and fitted the plane to the results extent.

The terrain did not. It was still the M0 load path from M2:

```gdscript
var tile: Dictionary = manifest["tiles"][0]     # data/tiles/demo/tiles.json
```

— tile **0** of a possibly unrelated tile set, at that set's `dx`. Through M5 that
was right by accident, because the run domain *was* the first tile. At reach scale it
put a 76.8 km water sheet over a 28.8 km patch of the Smoky Mountains DEM, so the
composite looked broken while both halves were correct. It was recorded in CLAUDE.md
as a gotcha precisely because verified-water and aligned-with-terrain are separate
claims and a screenshot merges them.

The same wrong bed also reached the **shader**: `water_surface.gdshader` reconstructs
`η = bed + depth` from `bed_tex`, so the surface was being lifted over elevations the
solver never used — not just a mis-drawn backdrop.

## 2. The fix: ship the bed with the frames

The viewer does not reassemble the mosaic. The solver already holds the one surface
that registers with the depth field cell for cell — the canonical store's `bed`
(§7.2), which is mosaic-assembled, windowed, coarsened, gap-filled and datum-un-shifted
— so `viewer_export` exports it into `frames/` as `manifest["static"]`, through the
**same tile layout and the same entry shape as a frame**. One decoder reads both
(`_read_field_image`), and terrain and water share extent, origin and cell size **by
construction** rather than by two implementations agreeing.

The alternative — teaching GDScript to read `tiles.json`, place tiles, block-mean
coarsen and fill gaps — is a second implementation of `solver/io/mosaic.py` +
`coarsen.py` in a language with no tests, and it would still not know the window.

**Cost:** one extra `.raw` per tile per run (4 × 1 MB on the reach demo), written once,
read once at load. The frames themselves are unchanged, byte for byte.

### What changed

- **`solver/io/viewer_export.py`** — `_write_field` now writes any `(ny, nx)` field
  either whole or through the tile layout, and frames and the bed both go through it.
  `manifest["static"]` carries `files`/`tiles` plus the bed's min/max;
  `manifest["domain"]` copies the mosaic assembly record from `.zattrs` (origin, tiles
  used, **gap cells, fill value**) and `manifest["coarsen"]` the resolution choice.
  A store without a `bed` array simply gets no `static` section.
- **`viewer/scripts/results_player.gd`** — terrain is now built **from results** and
  is **re-entrant**: `_apply_geometry(bed, w, h, dx)` is the single place terrain,
  water plane, `bed_tex` and camera are fitted, so they cannot end up on different
  extents. It runs on load *and* when a run finishes (which can change the grid under
  a live viewer), rebuilding the Terrain3D node and clearing its region directory
  first — regions are keyed by world position, so importing a smaller domain over a
  larger one would leave the difference standing.
- The M0 tile stays as the **fallback**: the pre-run scene ("no results yet — press
  Run solver") and any store exported before the bed shipped. A fallback that does not
  cover the run warns rather than rendering silently.
- `terrain_view.tscn` / `terrain_loader.gd` (the M0 standalone scene) are unchanged.
  Tile 0 is *correct* there: an M0 tile set has no run to define a window, so there is
  no right mosaic to assemble.

### Camera clip planes, found by rendering

The first mosaic render came back **empty sky**, with Godot's light culler spewing
`create_frustum_points` failures: a fixed `far = 4e6` against a `near` of 0.05 is a
degenerate depth range. Clip planes now scale with the domain (`near = dx`,
`far = 4 × span`). This is why the gate is a screenshot and not only `--rbverify` —
the headless check passed on the frame that rendered nothing.

## 3. Validation

- **`--rbverify` now gates registration.** It previously asserted frames-loaded +
  water-visible, both of which were true of the broken composite. It now also asserts
  that the terrain's cell count and cell size equal the results grid's **and that the
  imported surface is the exported bed** — Terrain3D is sampled and its range bracketed
  against `manifest["static"]["bed"]`, which catches a stale or partial import, not just
  an absent one:

  ```
  terrain=768x768 @ 100.00m sampled 86.4..260.0 m vs bed 85.0..260.0 m, run_bed=true
  ```

  The sampling is deliberate: `get_height_range().x` reads `0` off an uninitialised
  region-padding texel (a known Terrain3D quirk, already noted in `terrain_loader.gd`),
  so any "relief" derived from it is really the *maximum elevation* — 260 m on a basin
  whose true relief is 175 m. **The gate has teeth**: strip `static` from a manifest and
  the tile-0 fallback fails it, exit 1, because a 1024² @ 28.15 m terrain under a
  768² @ 100 m run *is* non-registration. That is the intended verdict for a legacy
  export, not a regression — re-export the frames.
- **`reach_basin.toml` on the 5090** (the mosaic scenario, re-run): mass **1.60e-07**,
  unchanged from the M6 sign-off, and `--rbshot` renders the whole 76.8 km basin with
  the channel traced down the valley and the storm sheet across it — the water ends
  exactly at the terrain edge.
- **M2 demo regression**: re-exported and re-shot, the same composition as the stored
  `results_screenshot.png` (1024² @ 28.15 m — the only difference in the frame is the
  depth readout, 10.49 → 10.48 m, which is the precision pass, not this change: that
  screenshot predates it). `--rblaunch` drives the
  full subprocess loop to `done` at **2.59e-08** — the precision-pass figure, unchanged
  — with the terrain rebuilding after the run finishes (the re-entrant path, live).
- **241 tests green**, including three new export tests: the bed round-trips the store
  byte for byte, tiles with the frame layout, and the manifest carries the assembly
  record.

## 4. Carried limitation — **closed 2026-08-17, and this section was wrong**

**The channel surface is still the floodplain approximation.** The shader lifts water
to `η = bed + depth`, which is M1's storage relation. With M6 sub-grid channels a
channel cell's true surface below bank full is `η = z − d + h·dx/w`
(`solver/core/channels.py`), so the rendered sheet sits up to the channel depth `d`
high along the river. Terrain and water now register; the *channel* does not yet
render its own storage curve.

That much held — measured on this very demo, the drawn surface was up to **2.74 m** high
over a wet channel cell, and it arched *up* over the valley, drawing the river as a ridge.
What did not hold is the fix this section then prescribed: *"exporting
`channel_width`/`channel_depth` alongside the bed and evaluating the same curve in the
shader — a small job"*. **Evaluating the whole curve hides the river inside the ground.**
Below bank full the true surface is *under* the floodplain bed — up to **2.46 m** under
it, on **1030 of 2232** channel cells here — and the rendered terrain has no trench to
put it in, because the channel is sub-grid and a per-cell height map cannot hold a
feature narrower than a cell. Carving one is a whole-cell groove up to **14.7×** too
wide, below the water mesh's resolution, and it costs the "terrain *is* the exported bed"
invariant this pass bought.

What shipped instead: the fields do travel through `manifest["static"]`, and the shader
takes the exact curve **overbank** (where the old lift was wrong by `h_bf`, up to 1.39 m)
and draws the **bank** below it — never above the terrain, never buried under it, with
the one-sided residual measured per run and declared in the manifest. See
`viewer-channel-surface.md`.
