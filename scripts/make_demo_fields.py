"""Generate demo parameter-field .r32 rasters aligned to the M0 demo tile (M3).

The M3 solver accepts spatially-varying ``manning_n`` / ``infiltration`` /
``rainfall`` fields as raw little-endian float32 ``.r32`` matching the terrain
tile exactly (HANDOFF §7.1; see ``solver/io/fields.py``). This script derives a
couple of physically-motivated fields from the demo bed so the field code path
can be exercised end-to-end without hand-authoring rasters:

  * ``manning.r32``   -- roughness that rises with elevation (rougher upland
    vegetation, smoother valley floors);
  * ``infil.r32``     -- infiltration (mm/hr) that is higher on lowland soils and
    near-zero on steep upland rock.

Output (gitignored ``data/fields/``) is consumed by ``scenarios/spatial_fields.toml``.

Run: uv run python scripts/make_demo_fields.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TILES = REPO / "data" / "tiles" / "demo"
OUT = REPO / "data" / "fields"


def main() -> None:
    manifest = json.loads((TILES / "tiles.json").read_text())
    t0 = manifest["tiles"][0]
    h, w = int(t0["height"]), int(t0["width"])
    bed = np.fromfile(TILES / t0["file"], dtype="<f4", count=h * w).reshape(h, w)

    # Normalise elevation to [0, 1] for blending upland vs lowland properties.
    z0, z1 = float(np.nanmin(bed)), float(np.nanmax(bed))
    zn = np.clip((bed - z0) / max(z1 - z0, 1e-6), 0.0, 1.0).astype(np.float32)

    # Manning n: 0.030 (smooth valley) -> 0.070 (rough upland).
    manning = (0.030 + 0.040 * zn).astype(np.float32)
    # Infiltration mm/hr: 8 (lowland soils) -> ~0.5 (steep upland rock).
    infil = (8.0 * (1.0 - zn) + 0.5).astype(np.float32)

    OUT.mkdir(parents=True, exist_ok=True)
    manning.tofile(OUT / "manning.r32")
    infil.tofile(OUT / "infil.r32")
    print(
        f"wrote {OUT / 'manning.r32'}  ({w}x{h}, n in [{manning.min():.3f}, {manning.max():.3f}])"
    )
    print(f"wrote {OUT / 'infil.r32'}  ({w}x{h}, mm/hr in [{infil.min():.2f}, {infil.max():.2f}])")


if __name__ == "__main__":
    main()
