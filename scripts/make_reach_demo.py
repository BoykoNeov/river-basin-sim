"""Generate the synthetic multi-tile basin for the M6 reach demo scenario.

``scenarios/reach_basin.toml`` needs three things at once, none of which the M0
demo tile provides: a **tile set** (so the mosaic assembly is exercised, not just
tile 0), terrain whose river is **narrower than a cell** (so the sub-grid channel
is doing real work), and channel fields aligned to that mosaic. Like
``make_reservoir_demo.py``, this writes synthetic data whose numbers are checkable
by eye rather than shipping a large DEM.

Geometry (6 x 6 tiles of 256 cells at 50 m -> 1536 x 1536 = 76.8 km square):

  * the valley falls 0.075 m per row southward (a 0.15% slope, 115 m over the
    domain) toward an open southern boundary;
  * a **sinuous river** meanders about mid-column with a 3-cell-wide band of
    channel cells -- 250 m of *cells* carrying a river that is 7-45 m wide;
  * the floodplain rises 0.1 m per cell away from the river out to 120 cells, then
    0.5 m per cell into the valley walls (capped at 60 m), so water that leaves the
    channel spreads but does not escape the basin;
  * upstream drainage area grows 20 -> 300 km² down the reach, and the channel
    width/depth come from :func:`pipeline.channels.hydraulic_geometry` with
    demo-calibrated coefficients (a real basin needs its own -- that function's
    docstring says why).

The scenario then runs the mosaic at ``coarsen = 2`` -- 768 x 768 cells at 100 m, a
quarter of the cells -- with the river carried sub-grid, which is the whole point
of M6: the run is coarse, the river is not lost. At that size the per-frame viewer
export also crosses its 512-tile threshold, so the demo exercises §7.3's tile grid
as well.

Outputs (gitignored):
  ``data/tiles/reach_demo/``   4 x ``.r32`` + ``tiles.json``   (the domain)
  ``data/fields/reach_demo/``  ``channel_width.r32`` + ``channel_depth.r32``

Run: uv run python scripts/make_reach_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.channels import hydraulic_geometry  # noqa: E402

TILES_OUT = REPO / "data" / "tiles" / "reach_demo"
FIELDS_OUT = REPO / "data" / "fields" / "reach_demo"

TILE = 256  # cells per tile side
NTILE = 6  # tiles per side -> 1536 x 1536 cells
DX = 50.0  # metres
NY = NX = TILE * NTILE

# Demo-calibrated downstream hydraulic geometry (regional inputs, not constants --
# see pipeline/channels.py). Chosen so the river stays narrower than a 50 m cell.
WIDTH_COEF, WIDTH_EXP = 1.5, 0.5
DEPTH_COEF, DEPTH_EXP = 0.4, 0.3
AREA_HEAD_KM2, AREA_OUTLET_KM2 = 20.0, 900.0
# The channel band must be wide enough that consecutive rows *share* a column:
# the meander drifts up to ~1.3 cells per row, and a band that skips leaves a face
# with min(w) = 0, i.e. a river cut in half. `check_continuity` proves it, because
# this is exactly the kind of break that looks fine in a plot.
RIVER_HALF_CELLS = 2


def river_column(rows: np.ndarray) -> np.ndarray:
    """Meandering centreline column for each row (cells)."""
    return (NX - 1) / 2.0 + 70.0 * np.sin(2.0 * np.pi * rows / 340.0)


def basin_dem() -> tuple[np.ndarray, np.ndarray]:
    """Bed elevations (m) and upstream drainage area (km²) for the demo basin."""
    rows = np.arange(NY, dtype=np.float64)[:, None]
    cols = np.arange(NX, dtype=np.float64)[None, :]
    centre = river_column(rows)
    off = np.abs(cols - centre)

    valley = 200.0 - 0.075 * rows  # 0.15% southward slope
    # Walls are capped at 60 m: a basin with kilometres of relief and a
    # millimetre-thin rain sheet is the float32 eta = h + z conditioning trap
    # (CLAUDE.md), and a demo should not sit on the edge of the mass gate for
    # arithmetic reasons that have nothing to do with what it is demonstrating.
    flanks = np.minimum(np.where(off <= 120.0, 0.1 * off, 12.0 + 0.5 * (off - 120.0)), 60.0)
    bed = (valley + flanks).astype(np.float32)

    # Drainage area grows downstream, and only the river band carries a channel.
    grow = AREA_HEAD_KM2 + (AREA_OUTLET_KM2 - AREA_HEAD_KM2) * (rows / (NY - 1))
    area = np.where(off <= RIVER_HALF_CELLS, grow, 0.0)
    return bed, area


def check_continuity(width: np.ndarray) -> int:
    """Count row pairs whose channel bands do not share a column (0 = connected).

    A sub-grid channel conveys through a face only where *both* cells have one
    (``w_face = min(w_L, w_R)``), so a band that steps sideways faster than it is
    wide is a river with a wall across it -- and nothing in the depth field would
    obviously say so.
    """
    has = width > 0.0
    return int(sum(1 for i in range(has.shape[0] - 1) if not np.any(has[i] & has[i + 1])))


def write_tiles(bed: np.ndarray) -> dict:
    """Cut the bed into an M0-format ``NTILE x NTILE`` tile set."""
    TILES_OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for r in range(NTILE):
        for c in range(NTILE):
            r0, c0 = r * TILE, c * TILE
            block = np.ascontiguousarray(bed[r0 : r0 + TILE, c0 : c0 + TILE], dtype="<f4")
            name = f"tile_{r:02d}_{c:02d}.r32"
            (TILES_OUT / name).write_bytes(block.tobytes(order="C"))
            entries.append({"file": name, "row": r0, "col": c0, "width": TILE, "height": TILE})
    manifest = {
        "dx_m": DX,
        "crs": "EPSG:32617",
        "tile_size": TILE,
        "dtype": "<f4",
        "layout": "row-major (C order), raw little-endian float32, metres",
        "height_min": float(bed.min()),
        "height_max": float(bed.max()),
        "tiles": entries,
        "source": "synthetic (scripts/make_reach_demo.py) -- M6 reach demo",
    }
    (TILES_OUT / "tiles.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    bed, area = basin_dem()
    manifest = write_tiles(bed)

    width, depth, clipped = hydraulic_geometry(
        area,
        dx=DX,
        width_coef=WIDTH_COEF,
        width_exp=WIDTH_EXP,
        depth_coef=DEPTH_COEF,
        depth_exp=DEPTH_EXP,
        min_area_km2=1.0,
    )
    FIELDS_OUT.mkdir(parents=True, exist_ok=True)
    width.astype("<f4").tofile(FIELDS_OUT / "channel_width.r32")
    depth.astype("<f4").tofile(FIELDS_OUT / "channel_depth.r32")

    river = width > 0
    broken = check_continuity(width)
    print(f"reach demo -> {TILES_OUT}")
    print(f"  domain      : {NY}x{NX} cells @ dx={DX:.0f} m ({NY * DX / 1000:.1f} km square)")
    print(f"  tiles       : {len(manifest['tiles'])} ({NTILE}x{NTILE} of {TILE})")
    print(f"  bed         : {bed.min():.1f}..{bed.max():.1f} m")
    print(
        f"  channel     : {int(river.sum())} cells, width {width[river].min():.1f}"
        f"..{width[river].max():.1f} m, depth up to {depth.max():.2f} m"
    )
    print(f"  fields      : {FIELDS_OUT}")
    print(f"  continuity  : {'connected' if broken == 0 else f'BROKEN at {broken} row pairs'}")
    if clipped:
        print(f"  NOTE: {clipped} cells clipped to dx -- the demo grid is coarse for this river")
    print("  next: uv run python -m solver.run --config scenarios/reach_basin.toml")


if __name__ == "__main__":
    main()
