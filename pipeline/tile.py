"""Tiling (M0).

Cut a conditioned DEM (see ``condition.py``) into engine-ready terrain tiles.

Each tile is written as **raw little-endian float32, row-major (C order)** -- the
same byte layout HANDOFF locks for viewer result tiles (HANDOFF §7) -- plus a
``tiles.json`` manifest describing the grid and per-tile georeferencing. One
artifact serves both consumers:

* the **solver** (M1+) memory-maps the ``.r32`` as its static bed field ``z``
  (metre elevations, no normalization, full float32 fidelity);
* the **viewer** builds a Godot ``FORMAT_RF`` heightmap ``Image`` straight from
  the same bytes -- no EXR/16-bit-PNG round-trip, so heights stay exact metres.

Tiles are square (default 1024, a Terrain3D region size). nodata cells (the
reprojection corner wedges) are replaced with the tile's minimum valid elevation
so terrain is flat there rather than spiking; the count is recorded per tile.

CLI::

    # full grid of 1024 tiles
    uv run python -m pipeline.tile --src data/dem/conditioned --out data/tiles/demo
    # single auto-picked tile centred on the main river (M0 demo)
    uv run python -m pipeline.tile --src data/dem/conditioned --out data/tiles/demo --single
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from pipeline.condition import NODATA

DEFAULT_TILE = 1024


def read_conditioned(cond_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict, rasterio.Affine]:
    """Load bed elevation + flow accumulation + metadata from a conditioned dir."""
    cond = Path(cond_dir)
    meta = json.loads((cond / "condition.json").read_text())
    with rasterio.open(cond / "bed_elevation.tif") as ds:
        bed = ds.read(1)
        transform = ds.transform
    with rasterio.open(cond / "flow_accumulation.tif") as ds:
        acc = ds.read(1)
    return bed, acc, meta, transform


def _tile_bounds(
    transform: rasterio.Affine, r0: int, c0: int, h: int, w: int
) -> tuple[float, float, float, float]:
    """CRS bounds (left, bottom, right, top) of the tile window at (r0, c0)."""
    left = transform.c + transform.a * c0
    top = transform.f + transform.e * r0
    right = left + transform.a * w
    bottom = top + transform.e * h
    return float(left), float(bottom), float(right), float(top)


def auto_window(bed: np.ndarray, acc: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left (row, col) of a clean ``size``-square window on a strong river.

    The global outlet (max accumulation) sits at the domain edge, exactly where
    the reprojection nodata wedges are -- a tile there is part flat slab. So we
    instead scan candidate windows on a coarse stride and pick the one with the
    strongest river (max accumulation) among those that are nodata-free, relaxing
    the nodata limit only if nothing qualifies.
    """
    h, w = acc.shape
    if size >= h or size >= w:
        return 0, 0
    stride = max(size // 4, 1)
    rows = list(range(0, h - size + 1, stride)) + [h - size]
    cols = list(range(0, w - size + 1, stride)) + [w - size]

    best = (0, 0)
    best_score = -1.0
    best_clean = False
    for r0 in rows:
        for c0 in cols:
            nodata_frac = float(np.mean(bed[r0 : r0 + size, c0 : c0 + size] == NODATA))
            river = float(acc[r0 : r0 + size, c0 : c0 + size].max())
            clean = nodata_frac <= 0.005
            # Prefer any clean window over any dirty one; within a class, max river.
            if (clean, river) > (best_clean, best_score):
                best, best_score, best_clean = (r0, c0), river, clean
    return best


def _write_tile(bed: np.ndarray, r0: int, c0: int, size: int, out: Path, name: str) -> dict:
    """Write one ``.r32`` tile and return its manifest entry."""
    window = bed[r0 : r0 + size, c0 : c0 + size]
    h, w = window.shape  # edge tiles may be smaller than `size`
    nodata_mask = window == NODATA
    valid = window[~nodata_mask]
    fill = float(valid.min()) if valid.size else 0.0
    filled = np.where(nodata_mask, fill, window).astype("<f4")  # little-endian float32

    path = out / f"{name}.r32"
    path.write_bytes(filled.tobytes(order="C"))
    return {
        "file": path.name,
        "row": r0,
        "col": c0,
        "width": w,
        "height": h,
        "h_min": float(valid.min()) if valid.size else 0.0,
        "h_max": float(valid.max()) if valid.size else 0.0,
        "nodata_cells": int(nodata_mask.sum()),
    }


def tile_dem(
    cond_dir: str | Path,
    out_dir: str | Path,
    size: int = DEFAULT_TILE,
    single: bool = False,
) -> dict:
    """Cut the conditioned bed into ``.r32`` tiles + write ``tiles.json``.

    ``single`` extracts one auto-picked tile centred on the main river (M0 demo);
    otherwise a full row-major grid of tiles is written (edge tiles are clipped).
    """
    bed, acc, meta, transform = read_conditioned(cond_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    height, width = bed.shape

    if single:
        r0, c0 = auto_window(bed, acc, size)
        origins = [(r0, c0)]
    else:
        origins = [(r, c) for r in range(0, height, size) for c in range(0, width, size)]

    valid_all = bed[bed != NODATA]
    tiles = []
    for r0, c0 in origins:
        gr, gc = r0 // size, c0 // size
        entry = _write_tile(bed, r0, c0, size, out, f"tile_{gr:02d}_{gc:02d}")
        entry["bounds"] = _tile_bounds(transform, r0, c0, entry["height"], entry["width"])
        tiles.append(entry)

    manifest = {
        "crs": meta["crs"],
        "dx_m": meta["dx_m"],
        "tile_size": size,
        "source_shape": [height, width],
        "source_bounds": meta["bounds"],
        "nodata": NODATA,
        "dtype": "<f4",
        "layout": "row-major (C order), raw little-endian float32, metres",
        "height_min": float(valid_all.min()),
        "height_max": float(valid_all.max()),
        "tiles": tiles,
    }
    (out / "tiles.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tile a conditioned DEM into engine-ready tiles.")
    p.add_argument("--src", required=True, help="conditioned dir (output of pipeline.condition)")
    p.add_argument("--out", default="data/tiles/demo", help="output tiles dir")
    p.add_argument("--size", type=int, default=DEFAULT_TILE, help="square tile size (cells)")
    p.add_argument("--single", action="store_true", help="extract one tile on the main river")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    m = tile_dem(args.src, args.out, args.size, args.single)
    print(f"tiled {args.src} -> {args.out}")
    print(f"  tile size  : {m['tile_size']} @ dx={m['dx_m']:.2f} m")
    print(f"  tiles      : {len(m['tiles'])}")
    print(f"  height     : {m['height_min']:.0f}..{m['height_max']:.0f} m")
    for t in m["tiles"][:4]:
        print(
            f"    {t['file']}: {t['width']}x{t['height']} "
            f"({t['h_min']:.0f}..{t['h_max']:.0f} m, nodata {t['nodata_cells']})"
        )
    if len(m["tiles"]) > 4:
        print(f"    ... (+{len(m['tiles']) - 4} more)")


if __name__ == "__main__":
    main()
