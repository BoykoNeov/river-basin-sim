"""Sub-grid channel geometry from flow accumulation (M6, plan §1.4).

The solver's sub-grid channels (:mod:`solver.core.channels`) are **data**: each
cell carries a channel width and a bank-full depth. This module produces those two
fields for real terrain, from the flow accumulation the M0 conditioning step
already computes, using **downstream hydraulic geometry** -- the empirical
observation that a river's width and depth scale with a power of its drainage
area::

    w = a_w · A^b_w        d = a_d · A^b_d        (A = upstream area, km²)

applied only where ``A >= min_area_km2``; elsewhere there is no channel
(``w = d = 0``) and the cell is pure floodplain.

**These coefficients are regional calibration inputs, not constants of nature.**
The defaults below are a humid-temperate starting point and will be wrong -- by a
factor, not a percent -- for an arid basin, a bedrock gorge or a managed lowland
river. A coarse run is only as good as the channel geometry it is fed (M6 plan §3),
so calibrate them against surveyed cross-sections or a width product before
reporting numbers, and record what you used: the CLI writes the coefficients beside
the fields for exactly that reason.

Outputs are raw little-endian float32 ``.r32`` aligned to the **tile mosaic** (the
run domain, :mod:`solver.io.mosaic`), which is the same field contract M3 uses for
roughness and infiltration.

CLI::

    uv run python -m pipeline.channels --src data/dem/conditioned \\
        --tiles data/tiles/demo --out data/fields/demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Humid-temperate downstream hydraulic geometry (see the module docstring: these
# are calibration inputs). w ~ 8 A^0.5 gives a 25 m channel at 10 km2 and an 80 m
# channel at 100 km2; d ~ 0.27 A^0.3 gives 1.1 m and 2.1 m at the same areas.
DEFAULT_WIDTH_COEF = 8.0
DEFAULT_WIDTH_EXP = 0.5
DEFAULT_DEPTH_COEF = 0.27
DEFAULT_DEPTH_EXP = 0.3
# Below this upstream area a cell is hillslope, not river.
DEFAULT_MIN_AREA_KM2 = 1.0


def hydraulic_geometry(
    area_km2: np.ndarray,
    *,
    dx: float,
    width_coef: float = DEFAULT_WIDTH_COEF,
    width_exp: float = DEFAULT_WIDTH_EXP,
    depth_coef: float = DEFAULT_DEPTH_COEF,
    depth_exp: float = DEFAULT_DEPTH_EXP,
    min_area_km2: float = DEFAULT_MIN_AREA_KM2,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Channel ``(width, depth, n_clipped)`` fields from upstream drainage area.

    Widths are clipped to ``dx``: a channel wider than its cell is not sub-grid,
    and the solver refuses it (:func:`solver.core.channels.validate_geometry`). The
    clip count is returned rather than swallowed -- a large one means the grid is
    too coarse for the river it is carrying, which is a modelling decision, not a
    detail.
    """
    area = np.asarray(area_km2, dtype=np.float64)
    river = area >= float(min_area_km2)
    width = np.where(river, width_coef * np.power(np.maximum(area, 0.0), width_exp), 0.0)
    depth = np.where(river, depth_coef * np.power(np.maximum(area, 0.0), depth_exp), 0.0)
    clipped = int(np.count_nonzero(width > dx))
    width = np.minimum(width, dx)
    # "No channel" must be zero in *both* fields (the solver's single test).
    depth = np.where(width > 0.0, depth, 0.0)
    return width.astype(np.float32), depth.astype(np.float32), clipped


def area_km2_from_accumulation(acc_cells: np.ndarray, dx: float) -> np.ndarray:
    """Upstream drainage area (km²) from a D8 flow-accumulation raster (cells)."""
    return np.asarray(acc_cells, dtype=np.float64) * (dx * dx) / 1.0e6


def _mosaic_window(tiles: list[dict]) -> tuple[int, int, int, int]:
    """Half-open ``(r0, c0, r1, c1)`` source-raster window covering a tile set."""
    r0 = min(int(t["row"]) for t in tiles)
    c0 = min(int(t["col"]) for t in tiles)
    r1 = max(int(t["row"]) + int(t["height"]) for t in tiles)
    c1 = max(int(t["col"]) + int(t["width"]) for t in tiles)
    return r0, c0, r1, c1


def channel_fields(
    cond_dir: str | Path,
    tiles_dir: str | Path,
    out_dir: str | Path,
    **coeffs: float,
) -> dict:
    """Write ``channel_width.r32`` / ``channel_depth.r32`` for a tile mosaic.

    Reads the conditioned flow accumulation (``pipeline.condition`` output), cuts
    the window the tile set covers, applies :func:`hydraulic_geometry`, and writes
    the two fields plus a ``channels.json`` note recording the coefficients used.
    """
    from pipeline.tile import read_conditioned  # local: needs rasterio (geo extra)

    _, acc, meta, _ = read_conditioned(cond_dir)
    manifest = json.loads((Path(tiles_dir) / "tiles.json").read_text())
    dx = float(manifest.get("dx_m", meta["dx_m"]))
    r0, c0, r1, c1 = _mosaic_window(manifest["tiles"])
    window = acc[r0:r1, c0:c1]

    width, depth, clipped = hydraulic_geometry(
        area_km2_from_accumulation(window, dx), dx=dx, **coeffs
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    width.astype("<f4").tofile(out / "channel_width.r32")
    depth.astype("<f4").tofile(out / "channel_depth.r32")
    record = {
        "source": str(Path(cond_dir)),
        "tiles": str(Path(tiles_dir)),
        "shape": [int(r1 - r0), int(c1 - c0)],
        "dx_m": dx,
        "coefficients": {
            "width_coef": coeffs.get("width_coef", DEFAULT_WIDTH_COEF),
            "width_exp": coeffs.get("width_exp", DEFAULT_WIDTH_EXP),
            "depth_coef": coeffs.get("depth_coef", DEFAULT_DEPTH_COEF),
            "depth_exp": coeffs.get("depth_exp", DEFAULT_DEPTH_EXP),
            "min_area_km2": coeffs.get("min_area_km2", DEFAULT_MIN_AREA_KM2),
        },
        "note": "downstream hydraulic geometry -- REGIONAL CALIBRATION INPUTS, not constants",
        "channel_cells": int(np.count_nonzero(width)),
        "width_clipped_to_dx": clipped,
        "width_max_m": float(width.max()),
        "depth_max_m": float(depth.max()),
    }
    (out / "channels.json").write_text(json.dumps(record, indent=2))
    return record


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sub-grid channel fields from flow accumulation.")
    p.add_argument("--src", required=True, help="conditioned dir (pipeline.condition output)")
    p.add_argument("--tiles", required=True, help="tiles dir (tiles.json) the fields align to")
    p.add_argument("--out", required=True, help="output dir for the .r32 fields")
    p.add_argument("--width-coef", type=float, default=DEFAULT_WIDTH_COEF)
    p.add_argument("--width-exp", type=float, default=DEFAULT_WIDTH_EXP)
    p.add_argument("--depth-coef", type=float, default=DEFAULT_DEPTH_COEF)
    p.add_argument("--depth-exp", type=float, default=DEFAULT_DEPTH_EXP)
    p.add_argument("--min-area-km2", type=float, default=DEFAULT_MIN_AREA_KM2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    a = _parse_args(argv)
    rec = channel_fields(
        a.src,
        a.tiles,
        a.out,
        width_coef=a.width_coef,
        width_exp=a.width_exp,
        depth_coef=a.depth_coef,
        depth_exp=a.depth_exp,
        min_area_km2=a.min_area_km2,
    )
    print(f"channel fields -> {a.out}")
    print(f"  grid        : {rec['shape'][0]}x{rec['shape'][1]} @ dx={rec['dx_m']:.2f} m")
    print(f"  channel     : {rec['channel_cells']} cells, width <= {rec['width_max_m']:.1f} m")
    if rec["width_clipped_to_dx"]:
        print(
            f"  NOTE: {rec['width_clipped_to_dx']} cells had a channel wider than dx and were "
            "clipped -- the grid is coarse for this river"
        )
    print("  coefficients are regional calibration inputs; see channels.json")


if __name__ == "__main__":
    main()
