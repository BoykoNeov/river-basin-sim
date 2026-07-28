# Scenarios

Example scenario configs (TOML) and saved command logs. A scenario is
**config + parameter fields + command log**, which together fully determine a run —
this is the reproducibility and sharing story (HANDOFF §7.1, §7.4). Every run also
writes a `<store>.provenance.json` sidecar recording the config + field hashes.

## Shipped examples

| File | Exercises |
|---|---|
| `demo_basin_rain.toml` | M2: uniform rain, closed boundaries, scalar Manning. |
| `river_reach.toml` | M3: an **inflow hydrograph** + scalar **infiltration** + an **open** (free-outflow) boundary, on the demo tile. Self-contained. |
| `spatial_fields.toml` | M3: spatially-varying **Manning** and **infiltration** `.r32` fields (generate them first with `scripts/make_demo_fields.py`). |
| `river_reach_hllc.toml` | M4: the well-balanced **HLLC** finite-volume scheme on the *same* setup as `river_reach.toml` — only `[meta].scheme` and the CFL differ, so the pair is a like-for-like LI-vs-HLLC side-by-side. |
| `reservoir_release.toml` | M5: a **dam** with a `target_stage` **release rule on the slow clock**, plus a tidal **`fixed_stage`** boundary and an inflow hydrograph. Needs its synthetic valley tile first: `uv run python scripts/make_reservoir_demo.py`. |
| `reach_basin.toml` | M6: **reach** — a 6×6 **tile mosaic** (1536² @ 50 m = 76.8 km square) run at `coarsen = 2` (768² @ 100 m) with the river carried by **sub-grid channels**. Needs its synthetic basin + channel fields first: `uv run python scripts/make_reach_demo.py`. |

Run one with:

```
uv run python -m solver.run --config scenarios/<name>.toml
```

## Parameter fields (M3)

`manning_n`, `infiltration`, and `rainfall` (`type = "field"`) accept either a
scalar or a **path to a raw little-endian float32 `.r32`** — row-major `(y, x)`,
matching the terrain tile's `(ny, nx)` exactly (the M0 tile convention). Paths are
resolved relative to the TOML file's directory. A field whose size doesn't match
the tile is a hard error, never a silent resample. (A GeoTIFF `.tif` is also
accepted when `rasterio`/the `geo` extra is installed, resampled to the grid.)

`scripts/make_demo_fields.py` derives demo Manning/infiltration `.r32` rasters from
the demo bed as a worked example.

## Boundaries (M3, M5)

`[boundaries] default` plus optional per-edge `north/south/east/west`. An edge is
either a bare string — `"closed"` (reflective) or `"open"` (transmissive /
free-outflow) — or, from M5, a **`fixed_stage` table**, which carries a prescribed
water surface and so cannot be written as a plain string:

```toml
[boundaries]
default = "closed"
east  = "open"                                            # free outflow
south = { type = "fixed_stage", level = 10.35 }           # constant water level
west  = { type = "fixed_stage", stage = [[0.0, 9.7],      # or a curve in time
                                         [3600.0, 10.35],
                                         [10800.0, 9.7]] }
```

A stage curve is piecewise-linear and **held** at its end values outside the range
(a water level does not vanish the way a hydrograph does); its knots become
scheduler sync points, so no step straddles a change of slope. Water crosses a
stage edge in **both** directions and is mass-accounted either way.

`fixed_stage` requires `scheme = "hllc_fv"` and is rejected on `local_inertial` —
the staggered scheme has no boundary face to prescribe a surface on (M5 plan §1.4).
The `inflow` boundary *type* stays deferred: `[[inflow]]` cell sources already cover
prescribed discharge and their mass accounting is exact by construction.

## Structures and reservoir operations (M5)

Repeat `[[structures]]` tables. A structure is **barrier geometry plus a rule**: its
`cells` are raised to `crest_m`, so impoundment and overtopping are ordinary
shallow-water physics, and a `dam` may additionally carry a release rule evaluated
on its **own slow clock** (`interval_s`) while the flood scheme sub-cycles freely.

```toml
[[structures]]
name = "valley_dam"
type = "dam"                 # "dam" | "levee" (a levee is barrier geometry only)
crest_m = 78.0
cells = [[60, 50], [60, 51]] # or a single cell = [row, col]
release_rule = "target_stage"   # "none" | "fixed" | "target_stage"
target_stage_m = 75.0           # target_stage: draw the pool down toward this
release_max_m3_s = 60.0         # target_stage: the cap, reached at the crest
# release_m3_s = 12.0           # fixed: a constant discharge instead
pool = [45, 50, 59, 78]      # inclusive [row0, col0, row1, col1] the release draws from
outlet = [64, 64]            # cell the released water is delivered to
interval_s = 900.0           # slow-clock cadence (simulated seconds)
```

The transfer is internal and **mass-exact**: the withdrawal is capped by what the
pool holds, and the float32 rounding of the delivery is banked rather than lost. The
release history is written to the store's `.zattrs` as `reservoir_releases`.

## Vertical datum (M5)

`[grid] datum = "auto"` (⇒ `floor(min(bed))`) or an explicit reference elevation
steps the run in shifted coordinates, protecting float32 `eta = h + z` at a high
datum, and un-shifts the bed on the way out — the store always records true
elevations, with the offset in `.zattrs` as `datum_shift_m`. Structure crests and
stage levels shift with the bed. It is worth setting for terrain hundreds of metres
above the datum; at ~10 m it measurably buys nothing (see
`validation/test_ea_test1.py`).

## Domain, resolution and sub-grid channels (M6)

The domain is the **tile mosaic**, and the run resolution is a choice:

```toml
[grid]
tiles_dir = "data/tiles/reach_demo"
tiles = "all"              # "all" (the whole tile set, default) | "first" (tile 0)
window = [0, 0, 767, 767]  # optional inclusive [row0, col0, row1, col1], mosaic coords
coarsen = 2                # run k times coarser than the tiles: dx' = k*dx
```

Only the tiles a `window` touches are read, and cells no tile covers are filled
flat **and counted** — a hole in the terrain is reported at startup, never
discovered later in the depth field. Coarsening aggregates every input field once,
before stepping: the bed and the rate fields (Manning, infiltration, rainfall) by
block **mean** (volume-preserving), channel width/depth by block **max** (a river
crosses a block; averaging its width away is the failure this exists to prevent).
Partial blocks are cropped, not padded, and the trim is reported.

**Cell indices** (`[[inflow]] cell`, `[[structures]] cells`/`pool`/`outlet`) are
authored in the **assembled-domain frame** — after `window`, before `coarsen` — and
mapped by `i // k`; an index outside the resolved domain is a loud error.

A cell may carry a **channel narrower than itself**, which is what lets a coarse run
keep its river:

```toml
[channels]
width = "../data/fields/reach_demo/channel_width.r32"   # m, 0 < w <= dx (or a scalar)
depth = "../data/fields/reach_demo/channel_depth.r32"   # m below the floodplain bed
manning = 0.03                                          # default: the floodplain value
```

Water below bank full sits in the channel (its depth amplified by `dx/w`), and a
face carries a channel flow plus a floodplain flow. Sub-grid channels are
**`local_inertial`-only** and rejected on `hllc_fv` (M6 plan §0). Generate the two
fields from a conditioned DEM's flow accumulation with

```
uv run python -m pipeline.channels --src <conditioned> --tiles <tiles> --out <fields>
```

whose hydraulic-geometry coefficients are **regional calibration inputs, not
constants** — they are written beside the fields in `channels.json` for that reason,
and a coarse run is only as good as the channel geometry it is fed.

## Scheme selection (M4)

`[meta] scheme = "local_inertial"` (default, the M1 Bates scheme — permanent
coverage for lowland floodplains) or `"hllc_fv"` (the M4 well-balanced HLLC
finite-volume scheme — the fidelity option for shocks, transcritical flow, and
well-balanced wet/dry). Both honour the same config; only the CFL differs in
practice, since HLLC's bound is velocity-dependent — use `[run] cfl ≈ 0.45` for
HLLC against LI's `≈ 0.7`. The output store is scheme-agnostic, so the viewer
reads either unchanged.

## Inflow hydrographs (M3)

Repeat `[[inflow]]` tables, each a `cell = [row, col]` and a piecewise-linear
`hydrograph = [[t_s, Q_m3_s], ...]`. Discharge is injected as a cell source
(a river mouth entering the domain), zero-held outside the curve.

Spread a large discharge over several cells rather than one — a point source strong
enough to fill a reservoir will otherwise pond metres deep in its own cell.
