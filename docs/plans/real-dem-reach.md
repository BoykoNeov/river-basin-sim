# Real DEM, end to end — the reach machinery on terrain that was not built to suit it

**Status: steps 1–5 done, 2026-08-17; steps 6–7 blocked on a defect step 5 found.** The
mosaic is cut, the fields are derived and the scenario ran — and it **does not convey**:
`coarsen`'s block-mean bed aggregation is volume-preserving but not *descent*-preserving,
so at 112.59 m the chosen river climbs 1.17× as much as it descends and 4.19 M m³ ponds
five cells below the inlet with the mass gate reading clean. See §5.3. Not a milestone.
M0–M7 are signed off and every carried
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

**Chosen and measured in §5.2.1** (`--max-area-km2 auto`), where the second consequence
turns up: the tributaries were connected *through* the trunk, so dropping it shatters the
network — on the M0 tile 29 pieces become 69, and two thirds of the inlet-to-outlet route
stops being channel. The road not taken is a coarser run: this window's biggest river
needs `dx ≥ 289 m` to be genuinely sub-grid, which is `coarsen = 11` and an 85² grid of
340 m cells on 14 % slopes. Not built, and not close.

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
   the first run. The verification is `--inlet` / `--outlet` on `pipeline.channels`
   (§5.2.3), and it has two jobs: the inflow cells must be in the channel, and they must
   be on a piece that drains out of the domain rather than into one of the conditioning's
   dead ends — 14.5 % of this window's channel cells are on pieces that do not.
5. **Run it, on both backends.** Mass gate < 1e-6 is necessary and not sufficient — a
   dammed river passes it. The discriminating check is that water actually reaches the
   outlet.
6. **View it.** `--rbverify`, then a screenshot: the channel-surface pass established
   that `--rbverify` never looks at the water surface, so the picture is a separate gate.
7. **Write it up**, including everything in §1 and §4 that real terrain does to the
   model's assumptions.

## 5.1 Steps 1–2 as shipped (2026-08-17)

`pipeline/channels.py` gains `rook_connect`, `isolated_cells`, `components`,
`connectivity_report` and `d8_offsets`; `channel_fields` takes `coarsen` and `connect`;
the CLI takes `--coarsen` and `--no-connect`. **374 → 387 tests green.** The new gates
live in `pipeline/test_channels.py`, deliberately **not** behind the geo extra's
`importorskip` the way `test_pipeline.py` is — they gate a defect nothing else can see,
so they have to run in a bare `uv run pytest`.

**Measured on the real DEM** (M0 tile, 1024² @ 28.1 m, derived at `coarsen = 4`), the
same fields with and without the fix, reported both as authored and as the solver
actually consumes them after its own block-max coarsening:

| | channel cells | 4-conn components | largest | 8-conn | interior orphans |
|---|---|---|---|---|---|
| as-is, authored | 18 302 | **7 712** | 27 (0.1 %) | 29 | 3 969 |
| as-is, **as the solver runs it** | 5 918 | **435** | 151 (2.6 %) | 27 | 79 |
| fixed, authored | 25 962 | **29** | 19 934 (76.8 %) | 29 | **0** |
| fixed, **as the solver runs it** | 6 373 | **27** | 4 891 (76.7 %) | 27 | **0** |

The fixed network's rook-connected component count equals its 8-connected one at both
resolutions — the strongest form the gate can take — and coarsening does not undo it.

**Three things the implementation turned up that the plan did not predict.**

*A windowed domain legitimately has isolated cells.* After the fix exactly one cell on
the M0 tile has no 4-connected neighbour, and it sits on **row 0**: its river continues
outside the window. Counting that as a defect would make the gate unpassable for every
domain, since every domain is a window. `isolated_cells` grew an `interior_only` flag and
the report carries both numbers; the warning fires on the interior count.

*The clip count goes **up** when connectivity is fixed* — 1850 → 2740 cells on this tile
— because inserted cells inherit the through-river's width, so more cells carry a big
river. That is correct, and it would read as a regression to anyone who did not expect it.

*A float32 cast can undo the clip.* `validate_geometry` rejects on a strict `w > grid.dx`,
and the real coarsen-4 cell size 112.58551545578392 m has nearest float32
**112.58551788** — *larger*. Clipping in float64 and casting would hand the solver a
field it refuses, from data correct by construction. It passes today only because
`Grid.dx` is a Python float and NEP 50 therefore makes the comparison in float32; a
float64 `dx` off a manifest would turn that luck into an unexplained rejection. The clip
now steps down one ulp when the cast rounds up, and a test asserts the comparison the
solver actually makes. Verified end to end: the real derived fields, coarsened by the
solver's own `block_reduce` and passed to its own `validate_geometry`, are **accepted**,
with the widest channel at exactly `1.000000 × dx`.

**Not changed:** `scripts/make_reach_demo.py` has the same clip-resolution defect (it
passes the 50 m tile `dx` while `reach_basin` runs at 100 m), but its widths top out at
26 m so nothing is clipped at either resolution and every recorded demo figure is
unaffected. Left alone deliberately rather than fixed and re-measured.

## 5.2 The §4 cutoff, chosen — and the piece of river that goes nowhere (2026-08-17)

Two things were left after §5.1, and one of them turned out to be a question with no
answer as asked.

### 5.2.1 The cutoff is now a parameter, and it is written down

`subgrid_cutoff_km2(run_dx)` inverts the width law — `(dx / a_w)^(1/b_w)`, **198.1 km²**
at the coarsen-4 cell and 12.4 km² at the native one — and `--max-area-km2 auto` drops
the rivers above it from the channel mask. On the M0 tile at `coarsen = 4`:

| | channel cells | clipped to the cell | widest | pieces (4-conn) | largest piece |
|---|---|---|---|---|---|
| carried, clipped (the old default) | 25 962 | **2740** | 112.59 m = **1.000 dx** | 29 | 19 934 |
| cutoff `A ≤ 198.1 km²` | 23 222 | **0** | 108.97 m = 0.968 dx | **69** | 5220 |

The cells it drops are *exactly* the ones the clip was flattening — 2740 either way,
which is the invariant `pipeline/test_pipeline.py` gates — so this changes what is
carried, not how it is sized.

**The price is the shatter, and it is bigger than a component count suggests.** 29 → 69
pieces authored, 27 → 64 as the solver runs them after its own block-max coarsening, and
the largest piece falls from 19 934 cells to 5220. The reason is not subtle: those
tributaries were connected *through the trunk*, and the trunk is now floodplain. Measured
on the route actually chosen below, which is the form that matters: the flow path from
the inlet to the outlet is 396 cells either way, and **397 of 397 were channel before the
cutoff against 123 of 396 after it**. The water is put into a sub-grid channel and then
travels two thirds of the way to the outlet as ordinary on-grid flow.

**The field was handed to the code that consumes it**, which §5.1 established is where
the float32-ulp trap lives: `data/fields/smoky` block-maxed by `solver.io.coarsen` and
passed to `solver.core.channels.validate_geometry` is **accepted**, 5740 channel cells at
the run resolution (the figure `channels.json` now records beside the authored 23 222),
widest **0.968 dx**. One consequence to carry into step 4: the deepest channel left is
**1.29 m** bank full, because depth follows area too — with the cutoff on, the carried
rivers spill overbank early, and that is the model, not a bug.

That is the right trade at this resolution and it should still be said out loud. A
1262 km² river is 289 m wide — 2.6 cells at 112.59 m — so it does not need a sub-grid
model and cannot have one; carrying it clipped meant "the river is exactly one cell
across" on 2740 cells, precisely where the flood is. **Default off**: the cutoff is a
modelling choice, every figure recorded before this was measured without it, and the
module will not make the choice silently. The `NOTE` that fires on a nonzero clip count
used to point the wrong way — it said to *raise* `--min-area-km2`, which drops small
tributaries, keeps the trunk, and does not move the clip count by one cell.

### 5.2.2 The pruning cutoff cannot sever a river, and something else did

The 27 pieces invited the reading that some are one river cut in half where the
"too small to be a river" threshold pruned a link. **That cannot happen.** Flow
accumulation is monotone downstream, so once a cell is at or above `min_area` every cell
downstream of it is too: the derived mask is downstream-closed, and no choice of
`min_area` can open a gap in the middle of a flow path. Corner insertion is not needed
there because there was never a break there.

What actually breaks a piece off is the window edge — and one thing nobody had looked
for. The conditioned raster has **95 cells with no D8 direction at all** (pysheds writes
`-2` for a resolved outlet, `-1` for nodata, `0` for a pit), and the largest of them
carries **1262.5 km²**. It sits at field cell **[796, 869]**, in the middle of the M0
window, on a **292-cell flat at 530.5 m** (0.23 km², a 1.5 km reach SRTM renders as a
level water surface) that the fill left unresolved. Flow accumulation restarts at 1 below
it, so the derived river simply stops:

| | pieces | drain to a dead end **inside** the domain | cells stranded |
|---|---|---|---|
| carried, clipped | 29 | **2** | 3769 of 25 962 (**14.5 %**) |
| cutoff `A ≤ 198.1 km²` | 69 | **7** | 3459 |

The largest stranded piece is **3697 cells** — the whole south-east quarter of the
window's network — and water put into it ponds. This is the §2 defect's twin: mass is
conserved perfectly in a domain that fills instead of conveying, so **no gate in the repo
can see it**, and it survives every fix §2 made because it is a property of the terrain
conditioning, not of the mask. The cutoff neither causes nor cures it; it only re-cuts
the same stranded terrain into more pieces.

**A bounding-box touch is not a drainage test**, and believing it was would have hidden
this. All 27 pieces reach the window edge somewhere — including the stranded one, whose
tributaries run up to row 0. `drainage_check` traces the D8 path from each piece's own
**highest-accumulation cell** instead, and calls a piece sealed when that path dead-ends
away from the border.

**What a trace does and does not prove.** The direction raster comes from the *filled*
elevation while `pipeline.tile` writes the *raw* bed into the tiles the solver steps on,
so a trace is a statement about the derived network's routing, not a prediction about the
shallow-water run. Its negative is the useful half: a path ending with no direction well
inside the window is a stretch of river with nowhere to go. "Water reaches the outlet"
stays a run-time gate (§6) and nothing here stands in for it.

### 5.2.3 The route, checked

`route_report` answers three separate questions about a scenario's cells, given in the
field's own pre-coarsen coordinates — the same ones `[[inflow]] cell` uses. Is the inlet
in the channel at all (nothing else in this repo checks that, which is how `reach_basin`
came to inject onto floodplain two cells off its meander for two milestones); are inlet
and outlet in the same rook-connected piece (meaningful only with the main stem carried
as a channel — with the cutoff set the trunk between them is floodplain *by choice*); and
does the flow path from the inlet dead-end inside the domain.

The pair chosen for step 4, on `data/fields/smoky` (M0 tile, `coarsen = 4`, cutoff on):

- **inlet** — four consecutive flow-path cells `[266,327] [265,327] [264,327] [263,327]`,
  each 176.4 km² and 106.3 m wide: the largest river still carried sub-grid, and a run of
  four because a single-cell point source digs its own crater (M7 measured 83 % of a
  run's bed change under one).
- **outlet** — `[0, 172]`, where the trunk leaves the north edge, 396 cells downstream.
- Both are on the piece that drains out of the domain, and the stranded south-east
  quarter is avoided. Without the cutoff the same pair is in one piece of channel
  (19 934 cells); with it the outlet is floodplain, which the CLI says in those words
  rather than reporting an unanswered question as a failure.

One thing `route_report` deliberately refuses to answer: with inlets and no outlet there
is no pair, so `same_component` is `None` rather than `True` — four inflow cells in one
piece are not a route to anywhere, and reporting them as one would be a connection nobody
established. (It did, briefly. Advisor-caught.)

The census, the cutoff and the route are all written into `channels.json` beside the
coefficients. **387 → 401 tests green** — the connectivity, cutoff, trace and route gates
in `pipeline/test_channels.py` (pure numpy, no geo extra, because they gate defects
nothing else can see), the `channel_fields` plumbing in `pipeline/test_pipeline.py`.

## 5.3 Steps 3–5 on the full mosaic — the run that does not convey (2026-08-17)

The mosaic was cut, the fields were derived, the scenario was authored and it was run.
**Every gate §6 listed passed except the one that matters, and the reason is a design
choice M6 shipped, not anything in §2–§5.2.** This section is the finding; step 6
(viewer) and step 7 (final write-up) are not done, because what they would show is a
still pond.

### 5.3.1 The mosaic itself was uneventful, which is worth one paragraph

`pipeline.tile --src data/dem/conditioned --out data/tiles/smoky` writes **16 tiles**,
ragged edges of 919 rows and 211 cols, and the two bounding-box formulas
(`channels._mosaic_window`, `mosaic._bbox`) are byte-identical, so the shapes agree by
construction: **3991 × 3283, origin (0, 0), zero gap cells, 16/16 tiles used**, i.e.
997 × 820 = **817 540 cells** at `coarsen = 4`. The ragged-edge handling the mosaic
loader documents worked first time on real data. That was the whole of step 3's risk as
written, and it was not where the trouble was.

### 5.3.2 Three things the repo had written down were wrong

Found while reading the census, and all three are now corrected in the code:

**The pysheds codes were backwards.** `flowdir(..., flats=-1, pits=-2)`, verified from
the installed signature, with nodata at `0`. So **`-1` is an unresolved flat, `-2` is a
pit**, where this plan, `pipeline/channels.py` and CLAUDE.md all said "`-2` outlet, `-1`
nodata, `0` pit". All three assignments were wrong.

**The "292-cell flat at 530.5 m" is a single-cell pit.** Measured 8-connected on the
filled surface it is **1 cell**, its rim stands **1 mm** above it, and `fill_pits` raises
those cells by up to **1.28 m**. More usefully: on the **raw** bed *all 100* interior
no-direction cells have a lower neighbour, so they are **artefacts of the conditioning
chain, not closed basins in the terrain** — a different and more actionable claim than
the one recorded. They are also not cheaply fixable: `fill_pits` takes the stranded share
**39.1 % → 23.6 %**, a further `resolve_flats` pass to **20.1 %**. Iterating the
primitives converges slowly and not to zero, so this is documented, not fixed, and
`data/dem/conditioned` is deliberately **not** re-run — every recorded figure in the repo
depends on it.

**A one-tile census understated the strandings by 2.7×.** 39.1 % of the domain's valid
cells drain to one of 100 interior dead ends, against §5.2.2's **14.5 %** from the single
M0 tile. A window that happens to miss the pits looks clean.

Also recorded: **`filled_elevation.tif` is a diagnostic surface, not a reproducible
input.** `resolve_flats` works in float64 and separates a flat by a sub-millimetre
gradient that `write_outputs` casts to float32. Re-deriving directions from the written
file yields **376 345** flats against the shipped **14 096** — 27×. It cost an invalid
experiment here before it was noticed.

### 5.3.3 The connectivity warning was blaming the cutoff's price on the D8 defect

With the cutoff on the mosaic reads **458** components 4-connected against **456**
8-connected with **9** isolated cells, and the CLI printed §2's words: *"this network
fills rather than conveys — and the mass gate cannot see it."* The same network with the
cutoff **off** reads **163 / 163, 0 isolated**. So the gate breaks because the cutoff cut
tributaries loose from a trunk that is floodplain *by choice*, exactly as §5.2.1
predicted — and a reader was being sent after a bug that is not there.

`rook_connect` cannot simply run after the cutoff: it needs the uncut network to know
what the through-river is. So the fix is in the reporting. `isolation_cause` decides by
**measuring** the same network without the cutoff (`connectivity_without_cutoff`, now in
`channels.json`) rather than assuming, and the D8 text still fires when it is true.

### 5.3.4 The run: 4.19 M m³ in, 0.000 m³ out

`scenarios/smoky_reach.toml` — whole mosaic, `coarsen = 4`, datum auto, no rain, a
120 m³/s flood wave split across four cells verified in-channel and on a draining piece,
open boundaries. 12 h, 25 frames, CUDA.

| | |
|---|---|
| mass balance | **6.35e-07** (gate 1e-6) — *passes* |
| inflow_cum | 4 190 400 m³ |
| **outflow_cum** | **0.000 m³** |
| ledger residual | 2.2 m³ |
| wet cells (>1 mm) | **56** of 817 540 |
| deepest cell | 12.88 m, five cells below the inlet |
| water at the outlet | **none, at any frame** |

**The mass gate passed a run that moved nothing**, which is precisely what §6 anticipated
("necessary and not sufficient — a dammed river passes it"). The residual is a near-static
pond over 43 200 s at a 747 m datum, i.e. the conditioning gauge CLAUDE.md describes; it
will need re-measuring on a run that conveys and should not be tuned against this one.

### 5.3.5 The cause: block-mean coarsening is volume-preserving, not descent-preserving

M6 aggregates the bed by block **mean** because that conserves floodplain storage. On the
synthetic demo the valley is deliberately wider than a cell, so the mean is nearly
harmless. **A real mountain river valley is narrower than a 112.59 m cell and meanders
inside the block**, so the mean replaces the valley floor with an average of floor and
valley wall — and the river stops running downhill. Ascent summed along the chosen route,
on the bed the solver steps:

| bed aggregation | run `dx` | net drop | ascent | uphill steps | worst rise |
|---|---|---|---|---|---|
| raw | 28.15 m | 267.3 m | **0.00 m** | 0 / 337 | 0.00 m |
| block mean, `k=2` | 56.29 m | 267.3 m | **143.04 m** | 61 / 227 | 7.27 m |
| block mean, `k=4` | 112.59 m | 269.9 m | **314.92 m** | 49 / 118 | 23.42 m |
| block min, `k=2` | 56.29 m | 267.3 m | 0.00 m | 0 / 227 | 0.00 m |
| block min, `k=4` | 112.59 m | 267.3 m | 0.00 m | 0 / 118 | 0.00 m |

At the resolution the plan chose, **the river climbs 1.17× as much as it descends**.

**It is not one valley.** Over 23 independently sampled routes (20–200 km² headwaters,
≥ 50 cells long), block mean:

| | mean ascent | median | worst | uphill-step share | routes with zero ascent |
|---|---|---|---|---|---|
| `k=1` | 0.0 m | 0.0 | 0.0 | 0.0 % | 14 of 23 |
| `k=2` | 154.3 m | 56.5 | 518.1 | 23.9 % | 0 of 23 |
| `k=4` | 431.1 m | 122.2 | 1544.1 | 38.6 % | 0 of 23 |

and on the 21 routes whose drop at that resolution is still ≥ 10 m, ascent exceeds drop
on **9 of 21** at `k=2` and **18 of 21** at `k=4` (median ratio 0.92 and 2.10). At `k=4`,
**2 of 23 routes end higher than they start**. Two reporting notes, both learned the hard
way: a ratio against a non-positive drop is not a number (unguarded it printed 9.8e10, so
it is now `None`), and the `k=1` row reads "0.0 mean ascent" while only 14 of 23 routes
are *exactly* zero — the residue is real and expected, because the path comes from the
**filled** raster while the elevation is read on the **raw** bed, so a filled pit shows as
a small rise.

**Block min preserves the descent exactly** at every resolution — and is not the fix.
It lowers the whole terrain and destroys the floodplain-storage property that made mean
the choice. The better shape is probably to carry the thalweg in the sub-grid channel
invert `z − d`, which already exists and is already sub-grid; that is design work with
its own validation and it is **not** attempted here.

### 5.3.6 The synthetic sign-offs stand, and this was checked rather than assumed

`reach_basin` and `reach_alluvial` run the synthetic basin at `coarsen = 2`. Measured on
its analytic centreline: ascent **7.87 m** against a **115.01 m** drop — a ratio of
**0.068**, against the real DEM's median **0.92** at the same `k`. So the effect exists in
kind wherever block mean is used, is **≈14× smaller in ratio terms** on terrain built
wide and smooth, and never comes close to preventing conveyance. M6 and M7 are not in
question.

### 5.3.7 What this changes about the gates

The route check *passed* and was not enough. In-channel, 4-connected, drains out of the
domain — all three are properties of the **filled** D8 raster, and the solver integrates
the **coarsened raw bed**. `descent_report` is the missing pre-flight: it walks the flow
path on the block-mean bed at the run's `coarsen` and reports ascent, uphill steps, worst
rise and net drop. `pipeline.channels --inlet` now runs it and warns, so this costs a
second instead of a 12-hour run.

One trap inside the fix, caught by its own output: the conditioned raster keeps nodata as
a **-32768 sentinel**, and `pipeline.tile` replaces it per tile with that tile's minimum.
A flow path never enters nodata, which makes it look safe — but a *block mean around* a
path cell does, and the first version read a **33 508 m** "net drop" on a route that falls
268 m. `fill_tile_nodata` reproduces the tiler's fill first.

**401 → 409 tests green.**

## 6. Gates

- Every cell of the derived channel mask has a 4-connected neighbour in the mask, and the
  mask's 4-connected component count equals its 8-connected one (step 1's test). **When a
  cutoff is set this can legitimately fail** — attribute it with `isolation_cause` before
  reading it as the D8 defect (§5.3.3).
- **The route runs downhill on the bed the solver steps, at the run's `coarsen`**
  (§5.3.5, `descent_report`). Not implied by any of the network checks, and the one that
  would have caught the failed run before it started.
- Clip count is reported against the run resolution (step 2's test).
- The cutoff that decides what counts as a river is recorded in `channels.json`, and with
  it set nothing is clipped to its cell (§5.2.1's test).
- The scenario's inflow cells are in the channel, and on a piece whose flow path leaves
  the domain rather than dead-ending inside it (§5.2.3). Both are properties of the
  fields, checkable before a step is taken.
- Water reaches the outlet — the check the mass gate cannot make, and the one no
  property of the fields can stand in for.
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
