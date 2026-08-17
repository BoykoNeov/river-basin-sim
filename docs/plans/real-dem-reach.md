# Real DEM, end to end — the reach machinery on terrain that was not built to suit it

**Status: planned, 2026-08-17.** Not a milestone. M0–M7 are signed off and every carried
item is closed; this is the first thing chosen after the roadmap ran out. It is the one
path the repo has never walked: **every reach-scale run in the repo is synthetic.**
`scripts/make_reach_demo.py` builds the basin, the river and the channel fields that
`reach_basin` and `reach_alluvial` use. The real conditioned DEM
(`data/dem/conditioned`, SRTM `N35W083`) has only ever driven the single-tile M1/M2 demo
— no mosaic, no coarsening, no channels. `pipeline/channels.py` exists to derive channel
fields from real flow accumulation and **has never been run**: nothing in the repo
invokes its CLI, and only its `hydraulic_geometry` helper is used, by the synthetic
generator and two unit tests.

A survey of the real DEM was run before this plan was written, because a plan for real
terrain built on guesses is worthless. It found one blocker that stops the path outright,
one defect that silently understates the river, and three collisions between real terrain
and the model's assumptions. **The blocker is the reason this is worth doing.**

---

## 1. The survey

`data/dem/conditioned`, SRTM `N35W083` reprojected to UTM 17N (EPSG:32617),
`dx = 28.146 m`, `3991 × 3283` = **13.1 M cells**, 112.3 × 92.4 km, elevation
**218 – 2027 m** (mean 747, relief 1809), nodata **3.00 %**.

**It is steep.** Max downslope gradient over the four cardinal neighbours:

| run `dx` | p50 | p90 | p99 | cells > 10 % | cells > 30 % |
|---|---|---|---|---|---|
| 28.1 m | 0.193 | 0.437 | 0.660 | 72.5 % | 28.2 % |
| 56.3 m | 0.171 | 0.416 | 0.629 | 67.1 % | 25.1 % |
| 112.6 m | 0.142 | 0.385 | 0.594 | 60.7 % | 20.5 % |
| 225.2 m | 0.103 | 0.331 | 0.567 | 50.8 % | 13.6 % |

Coarsening barely helps — the median cell is still a 14 % grade at 113 m. M1 disclaims
exactly this: on the steep tile "mass + spatial pattern are sound, but steep-cell
velocities are limiter-shaped, **not** validated LI hydraulics". That disclaimer now
applies to the majority of the domain, not a tail of it, and it is a **caveat on the
deliverable, not something this pass solves**.

**The river is large.** Flow accumulation tops out at 1 723 547 cells = **1365 km²**.
Under `pipeline/channels.py`'s humid-temperate defaults (`w = 8·A^0.5`, `d = 0.27·A^0.3`,
`min_area = 1 km²`) that is a **295.6 m** channel, on 220 115 channel cells with a median
width of 16.6 m.

**Window size does not reduce it.** The obvious hope — pick a smaller window and the
river gets smaller — is false, and measurably so. Drainage area is *upstream* area, so a
5 km window sitting on the main stem sees the same 1365 km² a 58 km window does:

| clean window | biggest river in it | widest channel | needs `dx ≥` |
|---|---|---|---|
| 192 cells (5.4 km) | 1361.8 km² | 295.2 m | 295 m |
| 512 cells (14.4 km) | 1306.2 km² | 289.1 m | 289 m |
| 1536 cells (43.2 km) | 1262.5 km² | 284.3 m | 284 m |
| 2048 cells (57.6 km) | 1262.5 km² | 284.3 m | 284 m |

Fitted over eight sizes, `A_max ∝ L^−0.04` — **flat**. (A square-law assumption would
give exponent 2.00; it was written down, measured, and is wrong. What sets the required
cell size is the biggest river the window *touches*, not how big the window is.) So
"choose a smaller domain" is not a lever, and the only levers are the drainage-area
cutoff and the run resolution.

**Where nodata is.** 3.00 % overall, concentrated in the reprojection corner wedges: by
1024-tile, 0.0 % in the interior four but **16.7 %** in the south-east tile.
`pipeline/tile.py::_write_tile` replaces nodata with the tile's own minimum elevation, so
those become flat slabs at the local floor — water will pond on them and the picture will
show a lake that is an artefact. Either window away from them or declare them.

---

## 2. The blocker: a D8 network is not 4-connected, and the solver's faces are

This is the finding, and it stops the path until it is fixed.

`pipeline/channels.py` derives the channel mask from D8 flow accumulation. A D8 flow path
steps to whichever of **eight** neighbours is steepest, so roughly half its steps are
diagonal. The solver's channel faces are N/S/E/W only — a staggered grid has no diagonal
face — and face width is `min(w_L, w_R)`, so **a channel that is only diagonally
continuous has a wall across it**, and nothing in the depth field says so. HANDOFF §12
carries this hazard in as many words ("a plausible-looking result can be a dammed one");
it has never fired because the only channel fields ever used were hand-authored
contiguous bands.

Measured on the candidate window (`r2304 c1536`, 1536² native), channel mask `A ≥ 1 km²`,
component finder validated against a hand-built diagonal first:

| run `dx` | channel cells | 4-connected components | largest | 8-connected components | largest |
|---|---|---|---|---|---|
| 28.1 m | 40 448 | **19 008** | 37 (0.1 %) | 61 | 9 820 (24.3 %) |
| 56.3 m | 24 842 | 4 512 | 93 (0.4 %) | 61 | 5 998 (24.1 %) |
| 112.6 m | 13 422 | 1 016 | 275 (2.0 %) | 56 | 3 249 (24.2 %) |
| 225.2 m | 6 814 | 243 | 450 (6.6 %) | 49 | 1 646 (24.2 %) |

Read the two halves together. Under 8-connectivity the network is a **coherent dendritic
tree** — 61 components, the largest a quarter of all channel cells (the rest are separate
basins and pruned low-order links, which is what a windowed network looks like). Under
the 4-connectivity the solver actually has, the *same* network is **19 008 fragments
averaging two cells each**. Coarsening does not rescue it: even at 225 m the largest
rook-connected run of river is 6.6 % of the channel cells.

Fed to the solver as-is, `pipeline/channels.py`'s output is not a river. It is a chain of
disconnected pools that will fill, not convey — and every gate in the repo would pass,
because mass is conserved in a dammed domain exactly as well as in a flowing one. **This
is the same class of defect as the scheduler ripple: the mass gate is blind to it.**

### 2.1 The fix, measured — and the half of it that is not obvious

Make the derived mask rook-connected: at each diagonal step of the flow path, also carry
the channel through one of the two corner cells (the one with the larger accumulation).
**48.3 %** of channel cells take a diagonal step, so this inserts 19 497 cells, **+46.8 %**
of the mask at native resolution. As connectivity it is a complete fix:

| | channel cells | 4-conn components | largest | 8-conn components | cells with no 4-neighbour |
|---|---|---|---|---|---|
| derived as-is | 40 448 | **19 008** | 37 (0.1 %) | 61 | 11 382 (**28.1 %**) |
| corner-inserted | 59 389 | **61** | 14 323 (24.1 %) | 61 | **0** |

The 4-connected structure becomes *exactly* the 8-connected one — 61 components either
way, same largest — which is the strongest form the gate can take.

**But inserting the cell is not enough, and this is the part worth knowing.** A width
comes from the cell's *own* drainage area, and a corner cell is not on the flow path — it
is a hillslope cell beside the river. Its accumulation is tiny:

| | value |
|---|---|
| median own drainage area of an inserted cell | **0.01 km²** |
| median drainage area of the river passing through | **4.50 km²** |
| median `w_own / w_through` | **0.036** |
| inserted cells below the 1 km² channel threshold at all | 18 974 of 19 497 (**97.3 %**) |
| inserted cells that would throttle the face below 25 % of the river | **94.8 %** |

Since `w_face = min(w_L, w_R)`, giving the inserted cell its own width replaces the wall
with a **pinhole** — a 3.6 % aperture at the median. Conveyance would still be broken; it
would merely stop being obviously broken.

So the fix is **insert the corner cell *and* give it the through-path's width**
(`max(own, through)`). That is a change to what `hydraulic_geometry` means: a cell's
channel width stops being a pure function of its own drainage area and becomes *the width
of the river that passes through it*, which is the more defensible reading anyway — a
river's size is set upstream, not by the cell it happens to occupy.

With the through-width carried, the fix **survives coarsening**, which is the third way it
could have failed (it is applied at 28 m; the run coarsens with block max afterwards):

| run `dx` | as-is: 4-conn comps / largest | fixed: 4-conn comps / largest | fixed: no 4-neighbour |
|---|---|---|---|
| 56.3 m | 4 512 / 0.4 % | **61** / 24.1 % | 1 |
| 112.6 m | 1 016 / 2.0 % | **56** / 24.1 % | 1 |
| 225.2 m | 243 / 6.6 % | **50** / 24.1 % | 0 |

At every resolution the fixed network's 4-connected component count equals its
8-connected one. **Nothing else in this plan is worth doing before this.**

---

## 3. The second defect: the width clip is taken at the wrong resolution

`channel_fields` clips channel width to `dx` — "a channel wider than its cell is not
sub-grid, and the solver refuses it" — and takes `dx` from `tiles.json`, i.e. the
**tile** resolution. But `[grid] coarsen = k` means the run steps at `k·dx`, and
`solver/io/coarsen.py` aggregates channel width by block **max**, which cannot recover a
width that was already clipped away.

So a run at `coarsen = 4` on this DEM would be handed a river clipped to **28.1 m** while
its cells are **112.6 m** — the main stem understated by up to **10.5×** in width, and
with it the conveyance the whole sub-grid channel model exists to provide. The clip count
is reported (`width_clipped_to_dx`) but against the wrong denominator, so it reads 30 %
when the run's own figure is 5.4 %:

| clip taken at | cells clipped | of 220 115 | widest / `dx` |
|---|---|---|---|
| 28.1 m (tile — what the code does) | 66 076 | 30.0 % | 10.5× |
| 56.3 m (`coarsen = 2`) | 32 831 | 14.9 % | 5.3× |
| 112.6 m (`coarsen = 4`) | 11 926 | 5.4 % | 2.6× |
| 225.2 m (`coarsen = 8`) | 1 724 | 0.8 % | 1.3× |

**The fix** is a `--coarsen` / `--run-dx` argument on the CLI, clipping and reporting
against the resolution the run will actually use, and recording it in `channels.json`
beside the coefficients. Cheap, and it changes what the fields mean.

---

## 4. The modelling choice this forces, and its consequence

Even clipped correctly, the main stem does not fit. At `coarsen = 4` (112.6 m) the
candidate window's biggest river is 515.9 km² = a **181.7 m** channel: still 1.6× its
cell. Carrying it means clipping it to the cell, at which point it is not sub-grid — the
model degenerates to "the river is exactly one cell wide" precisely where the flood is.

The honest choice is to **set the drainage-area cutoff so every carried channel is
genuinely sub-grid, and let the main stem be floodplain**. At `coarsen = 4` the cutoff
that keeps `w ≤ dx` is `A ≤ 198 km²`, and it costs little:

| run `dx` | sub-grid cutoff | river cells kept | dropped |
|---|---|---|---|
| 28.1 m | A ≤ 12.4 km² | 29 312 of 40 448 | 27.5 % |
| 56.3 m | A ≤ 49.5 km² | 21 275 of 24 842 | 14.4 % |
| 112.6 m | A ≤ 198.0 km² | 12 612 of 13 422 | **6.0 %** |
| 225.2 m | A ≤ 792.2 km² | 6 814 of 6 814 | 0 % |

The consequence must be stated wherever the result is: **the main stem is resolved on the
grid like any other terrain, and only the tributaries are carried sub-grid.** That is a
defensible model — it is what the resolution can support — but it is not what the
synthetic demos show, and the difference should not be discovered by a reader.

**Why the synthetic demos never hit this.** `make_reach_demo.py` calls the same
`hydraulic_geometry`, with `WIDTH_COEF = 1.5` against the module's own humid-temperate
default of **8.0** — 5.3× narrower — on a basin whose drainage area is authored to grow
only to 300 km² against the real 1365. Both levers were turned toward "sub-grid", and the
docstring says so ("demo-calibrated coefficients (a real basin needs its own)"). The
demos are not wrong; they simply never tested the case real coefficients on real
accumulation produce.

---

## 5. Build order

Each step is independently checkable, and the first two are the substance.

1. **Rook-connect the derived network** (`pipeline/channels.py`). Insert the corner cell
   at each diagonal step of the flow path, choosing the higher-accumulation of the two,
   **and carry the through-path's width onto it** (§2.1 — without that it is a pinhole,
   not a channel). Needs the flow *direction* raster, which `channel_fields` does not
   currently read. Gate — stated as a binary property, not a tolerance, so it cannot be
   calibrated by its own answer: **every cell in the derived mask has at least one
   4-connected neighbour in the mask**, and the mask's 4-connected component count equals
   its 8-connected one. Both were measured exactly satisfied by the fix and exactly
   violated without it (28.1 % isolated, 19 008 vs 61 components). On a CI-sized fixture,
   no GPU.
2. **Clip at the run resolution** (`pipeline/channels.py` CLI). `--coarsen` / `--run-dx`,
   clip and report against it, record it in `channels.json`. Gate: a unit test that the
   same accumulation at two coarsen factors yields different clip counts and the coarser
   one is smaller.
3. **Cut the mosaic and derive the fields.** `pipeline.tile` full-grid (16 tiles at 1024,
   ragged edges of 919 rows and 211 cols — the mosaic loader documents that it handles
   them, which this exercises for the first time on real data), then `pipeline.channels`
   with the chosen cutoff. **Check the shapes agree before running anything**:
   `channel_fields` sizes its output from the tile bounding box while the mosaic gap-fills
   and counts uncovered cells, and a size mismatch between the two surfaces as a corrupt
   field file, not as the geometry error it is.
4. **Author the scenario.** `scenarios/smoky_reach.toml` — the candidate window, a datum
   shift (mean bed 747 m; float32 `η = h + z` has 6.1e-5 m of resolution there),
   local-inertial, an inflow hydrograph **split across several consecutive channel cells
   and verified to be in the channel**, open outflow at the basin mouth. No sediment in
   the first run.
5. **Run it, on both backends.** Mass gate < 1e-6 is necessary and not sufficient — a
   dammed river passes it. The discriminating check is that water actually reaches the
   outlet.
6. **View it.** `--rbverify`, then a screenshot: the channel-surface pass established
   that `--rbverify` never looks at the water surface, so the picture is a separate gate.
7. **Write it up**, including everything in §1 and §4 that real terrain does to the
   model's assumptions.

## 6. Gates

- Every cell of the derived channel mask has a 4-connected neighbour in the mask, and the
  mask's 4-connected component count equals its 8-connected one (step 1's test).
- Clip count is reported against the run resolution (step 2's test).
- Water reaches the outlet — the check the mass gate cannot make.
- Global mass balance < 1e-6 on both backends, quoted with the backend.
- `--rbverify` green, plus a screenshot at river level.

## 7. Out of scope, deliberately

- **The steepness.** 60 % of cells over a 10 % grade at the run resolution is outside
  local-inertial's regime, and no cutoff or resolution choice fixes it. The deliverable is
  "the real-data path works and here is where real terrain violates the assumptions", not
  a validated flood forecast for the Smoky Mountains. **Say so wherever a number is
  quoted.** A gentler DEM would make a prettier result and would not test anything this
  one does not.
- **Sediment.** M7 is local-inertial-only and this run is already at the edge of that
  scheme's regime; adding morphology on top would produce numbers nobody could defend.
- **Recalibrating the hydraulic geometry coefficients.** They are regional calibration
  inputs, the module says so, and calibrating them needs surveyed cross-sections this
  project does not have. Use the defaults, record them, and do not report a width as
  though it were measured.
