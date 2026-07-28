"""Generate the synthetic valley tile for the M5 reservoir demo scenario.

``scenarios/reservoir_release.toml`` needs terrain with a *known* impoundable
valley: a dam is placed by cell index and by crest elevation, so a scenario shipped
against arbitrary real terrain would either flood nothing or flood everything
depending on where the line happened to land. This script writes a small synthetic
tile with elevations chosen so the scenario's numbers are checkable by eye -- the
same role ``make_demo_fields.py`` plays for the M3 spatial-field scenario.

Geometry (128 x 128 cells at 40 m -> 5.12 km square):

  * bed = ``100 - 0.5*row`` metres: a valley floor falling 0.5 m per row from
    100 m at the north edge to 36.5 m at the south;
  * plus ``0.625 * |col - 64|`` metres: V-shaped valley walls rising 40 m to each
    side, so water concentrates in a channel down the middle.

The scenario then puts a dam line across row 60 (floor 70.0 m) with a crest at
78.0 m, drawing the pool down toward 75.0 m on a slow clock, while a tidal
``fixed_stage`` boundary works the southern edge (floor 36.5 m).

Output (gitignored ``data/tiles/reservoir_demo/``) is an M0-format tile: a raw
little-endian float32 ``.r32`` plus a ``tiles.json`` manifest.

Run: uv run python scripts/make_reservoir_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "tiles" / "reservoir_demo"

NY = NX = 128
DX = 40.0
CREST_ROW = 60


def valley_dem(ny: int = NY, nx: int = NX) -> np.ndarray:
    """Bed elevations (m) for the demo valley -- see the module docstring."""
    rows = np.arange(ny, dtype=np.float64)[:, None]
    cols = np.arange(nx, dtype=np.float64)[None, :]
    floor = 100.0 - 0.5 * rows
    walls = 0.625 * np.abs(cols - (nx - 1) / 2.0)
    return (floor + walls).astype(np.float32)


def main() -> None:
    bed = valley_dem()
    OUT.mkdir(parents=True, exist_ok=True)
    bed.tofile(OUT / "tile_000.r32")
    manifest = {
        "dx_m": DX,
        "crs": "EPSG:32617",
        "tiles": [{"file": "tile_000.r32", "height": NY, "width": NX, "row": 0, "col": 0}],
        "source": "synthetic (scripts/make_reservoir_demo.py) -- M5 reservoir demo",
    }
    (OUT / "tiles.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {OUT / 'tile_000.r32'}  ({NX}x{NY} @ {DX:.0f} m)")
    print(f"  elevation range      : {bed.min():.2f} .. {bed.max():.2f} m")
    print(f"  channel floor, row {CREST_ROW}: {bed[CREST_ROW, NX // 2]:.2f} m  (dam line)")
    print(f"  channel floor, row  44: {bed[44, NX // 2]:.2f} m  (pool head)")
    print(f"  channel floor, south : {bed[NY - 1, NX // 2]:.2f} m  (tidal edge)")


if __name__ == "__main__":
    main()
