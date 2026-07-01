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

## Boundaries (M3)

`[boundaries] default` plus optional per-edge `north/south/east/west`, each
`"closed"` (reflective) or `"open"` (transmissive / free-outflow). Open edges route
water off the domain; the leaving volume is mass-accounted. `fixed_stage`/`inflow`
boundary *types* arrive in M4.

## Inflow hydrographs (M3)

Repeat `[[inflow]]` tables, each a `cell = [row, col]` and a piecewise-linear
`hydrograph = [[t_s, Q_m3_s], ...]`. Discharge is injected as a cell source
(a river mouth entering the domain), zero-held outside the curve.

Structures/reservoirs arrive at M5.
