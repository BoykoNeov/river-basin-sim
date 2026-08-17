# Viewer channel surface — the river stops being a ridge

**Status: done, 2026-08-17.** The last carried item in the repo: `viewer-terrain-mosaic.md`
§4 ("the channel surface is still the floodplain approximation"), and through it
`roadmap.md` → "Carried debts before M7" item 2 and HANDOFF §7.3's *Known gap*. Not a
milestone — a scoped fix to one seam, the water surface a sub-grid channel is drawn at.
**The fix that item specified does not work, and finding out why is most of what this
pass produced.**

---

## 1. The defect

M6 gave a cell a channel narrower than itself, and with it a storage curve
(`solver/core/channels.py`): below bank full all the water is in the channel, so the
surface is `η = z − d + h·dx/w`; above it the channel is full and the rest spreads,
`η = z + (h − h_bf)`, with `h_bf = w·d/dx`.

The shader knew none of that. `water_surface.gdshader` lifted every vertex as
`η = bed + depth` — M1's relation — because the frames stream shipped `bed` and nothing
else. Measured on the shipped M6 demo (`data/results/reach_basin.zarr`, 768² @ 100 m,
2232 channel cells, final frame):

| | |
|---|---|
| wet channel cells drawn too high | **2232 of 2232** |
| worst | **2.74 m** |
| mean over wet channel cells | **0.92 m** |
| widest `dx/w` | **14.7** |

The error is `d + h·(1 − dx/w)`: largest when the channel is nearly empty (the water is
deep inside a trench, the picture draws it at the bank plus `h`), smallest at bank full
where it is `h_bf`. Overbank it is exactly `h_bf`, up to **1.39 m** here.

The shape of the error matters more than its size. A transect across the river at row
526 (frame 24, water in-bank, `w = 37 m`, `d = 2.76 m`):

```
 col   h      h_bf    bed z    drawn (bed+h)   true η     drawn above true
 402   0.920  1.032   121.32   122.24          121.02     1.22 m
 403   0.954  1.033   121.14   122.10          120.93     1.17 m
 404   0.899  1.033   121.21   122.11          120.85     1.26 m
```

The bed dips into the valley and the drawn surface **arches back up over it**: the river
was rendered as a ridge about a metre proud of its own floodplain. Not "a bit high" — the
wrong sign of curvature, exactly where a viewer looks to find the river.

## 2. Why the specified fix is not renderable

`viewer-terrain-mosaic.md` §4 said the fix was to export `w`/`d` and "evaluate the same
curve in the shader — a small job". Evaluating the *whole* curve makes the picture worse,
and the measurement is unambiguous. Below bank full the true surface is **under the
floodplain bed**:

| | |
|---|---|
| wet in-bank cells, final frame | **1030 of 2232** |
| true surface below the rendered bed, worst | **2.46 m** |
| same over all 25 frames (frame 1, a barely-wet channel) | **3.06 m** on 2223 cells |

The rendered terrain is the floodplain bed — it has no trench, because the channel is
**sub-grid**: `w` is 7–45 m inside a 100 m cell, and a per-cell height map cannot hold a
feature narrower than a cell. Drawing the true elevation there puts the river *inside the
ground*: occluded, z-fighting where it grazes, and gone on the 1030 cells that most need
to read as a river.

So the choice is not "correct vs approximate". Both available renderings are approximate
and they err in opposite directions; the only question is which one lies legibly.

**Carving the trench was considered and rejected.** Render the terrain at `z − d` in
channel cells and the water at its true level does sit inside it — and buys three
problems: the groove is a whole cell wide, up to **14.7×** wider than the river it
depicts; the water mesh is coarser than the grid (`WATER_SEGMENTS = 512` over 768 cells),
so a one-cell groove is below what the geometry can resolve and renders as spikes and
cracks, dragging in mesh resolution and per-texture filtering; and the terrain would stop
being the exported bed, which is the invariant `viewer-terrain-mosaic.md` bought and
`--rbverify` asserts by bracketing the sampled surface against the exported bed's range.
That is a rendering project, not a seam fix.

## 3. What shipped

`viewer_export.render_eta` — the **renderable projection** of the storage curve, and the
reference the GLSL is written from:

```
no channel :  η = z + h                 the M1 relation, bit for bit
h >  h_bf  :  η = z + (h − h_bf)        the exact M6 curve  (was wrong by h_bf)
h <= h_bf  :  η = z                     the bank, flat      (true surface is below it)
```

Continuous at bank full (both branches give `z`), monotone in `h` so a rising flood never
steps, and **one-sided**: the drawn surface is never *below* the true one, which is what
lets a single number bound the residual. Overbank is now exact; in-bank is drawn at the
bank rather than in a trench that is not there.

- **`solver/io/viewer_export.py`** — `manifest["static"]` ships `channel_width` and
  `channel_depth` beside `bed` when the store carries them (§7.2), through the same
  `_write_field` layout and the same entry shape, so `_read_field_image` decodes them
  unchanged. `render_eta` is host numpy beside `eta_from_h`, and the export **measures
  its own residual** over every frame — `static.channel.in_bank_offset_m` with the cell
  count and the frame it occurred on. Over all frames, not the last: the worst in-bank
  cell is a barely-wet one and the final frame is not where the flood is (3.06 m at frame
  1 against 2.46 m at frame 24). A bound from `d` would have been free; this is a
  property of *this picture*, which is the same reason `morphology` ships the measured
  bed-change extremes rather than the alluvium thickness.
- **`viewer/shaders/water_surface.gdshader`** — the three branches above, behind
  `has_channels`. `w`/`d` sample with **`filter_nearest`**: channel presence is a genuine
  discontinuity in the parameterization and interpolating a 20 m width against a 0 m
  neighbour invents a river between them. `bank_bias = 0.05 m` is added to both channel
  branches — a depth-buffer offset, not physics, because an in-bank surface is coplanar
  with the terrain it is drawn on; constant, so continuity survives it, and `render_eta`
  does not model it.
- **`viewer/scripts/results_player.gd`** — `_set_channel_geometry` binds the pair (or 1×1
  zeros, so a channel-free run loaded after a channelled one cannot inherit its river —
  an *unbound* sampler is undefined, not off), `_apply_geometry` sets `cell_size` with
  the rest of the domain, and `_report_channel_surface` prints the declaration in the
  same idiom as gap fill and a stale morphed bed. `--rbverify` gained the one check that
  can fail here: the decoded channel-cell count must equal the export's own count. A
  failed tile read returns a blank of the *right size*, so shape proves nothing.

## 4. Verification

- **374 tests green** (368 → 374). Six new: the three properties of `render_eta` above
  against `channels.eta_from_h`; a channel-free cell drawn `z + h` bit for bit (including
  half a channel — width without depth must not select the curve); the static fields
  matching the store untiled and tiled; the manifest's residual reproduced by an
  independent sweep over all frames; and a channel-free run shipping no channel key and
  no channel file.
- **`--rbverify` on `reach_basin`: OK**, `channel_cells=2232/2232`, terrain 768² @ 100 m
  sampled 86.4..260.0 m against the exported bed's 85.0..260.0 m.
- **The picture.** `--rbverify` cannot see this class of defect — it never looks at the
  water surface — so the gate is a screenshot pair at a river-level camera (a scratch
  `RB_CAM_AT` override in `_fit_camera`, deliberately not committed). Before/after differ
  on **0.29 %** of pixels in the river view and **0.05 %** at whole-basin altitude, and
  the changed pixels trace the river margins and nothing else: the fix is where the
  channel is, the render is otherwise untouched, and no z-fighting speckle appeared along
  the bank. A 1–3 m offset over a 76.8 km scene is nearly invisible from the default
  camera, which is why this survived M6 and M7 — "no symptom" meant no instrument again.
  The transect in §1 is what makes it legible.
- **Full loop `--rblaunch` green** end to end on the demo, `mass_max_rel_err=2.30e-08` —
  the scheduler pass's figure, unchanged, and the Windows `os.replace` handoff race did
  not fire with two more static files in the export.
- **Channel-free runs are unchanged by construction**: the store has no channel arrays,
  so nothing extra is written and `has_channels` stays false, leaving the vertex
  arithmetic the pre-change expression. The demo's manifest is M6's key for key (tested),
  and its re-export renders identically.

## 5. Carried limitations

1. **In-bank water is drawn up to `d` above its true surface** — 3.06 m at worst on the
   M6 demo, ~0.3 m near bank full, and the run says so in the manifest and in the
   viewer's log. This is the deliberate residual of §2, not a defect to fix later at the
   same level: closing it means giving the *terrain* the channel, which means a height
   field finer than the channel is wide.
2. **Colour is still the cell-mean depth `h`**, the quantity the manifest's colormap range
   (global p99) is computed over. Over a channel cell that is volume spread across the
   whole cell, not the depth of the river — shallower by up to `dx/w`. Recolouring by the
   column depth needs its own measured range or every river cell saturates, so it is a
   separate pass with its own number, not a rider on this one.
3. **The water mesh is coarser than the grid** (512 segments over 768 cells), so a river
   one cell wide is not resolved by the *geometry* in either version. What this pass
   fixes is the surface over and beside it, which is what a reach-scale view shows.
4. **An older `frames/` export ships no channel geometry and nothing can warn about it.**
   The manifest is the only record the viewer has, and a manifest written before this
   pass has no `channel` key to be missing — so the shader falls back to `bed + depth`
   silently. Re-export (`python -m solver.io.viewer_export <zarr> <out_dir>`) rather than
   trusting an old frames directory; the same standing advice the bed already carries.
5. **Terrain animation for a morphing bed stays deferred** (M7 §1.7), and one of its two
   reasons is now gone: the lift is fixed, so what remains is the animation itself —
   re-fitting the height map, the water plane, `bed_tex` and the registration check per
   frame. The curve's other input is safe under morphology: M7 freezes the section, so
   Exner moves `z` and the invert `z − d` translates with it, leaving `d` valid.
