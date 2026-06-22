"""DEM conditioning (M0).

Turn a raw geographic DEM into engine-ready, hydrologically-conditioned rasters:

    reproject to a metric CRS  ->  sink-fill  ->  resolve flats
                               ->  D8 flow direction  ->  flow accumulation

Two elevation surfaces come out, and the distinction is deliberate:

* **bed elevation** -- the *true* DEM, reprojected to metres. This is the physical
  bed the shallow-water solver integrates over (M1+) and the surface the viewer
  renders. We do NOT hand the solver a sink-filled bed: filled pits are real
  depressions (ponds, quarries, closed basins); flattening them would erase
  physics the solver is supposed to resolve.
* **filled elevation** -- the hydrologically conditioned surface, used *only* to
  derive a continuous, pit-free flow network (direction + accumulation). High
  accumulation traces the river/drainage network.

The conditioning backend here is pysheds (see ``pipeline/_compat.py`` for the
NumPy-2.x shim it needs). The public entry point ``condition_dem`` returns a
plain ``ConditionedDEM`` dataclass and writes plain GeoTIFFs + JSON, so the
backend can be swapped for WhiteboxTools without touching callers or outputs.

CLI::

    uv run python -m pipeline.condition \
        --src data/dem/raw/N35W083.hgt \
        --out data/dem/conditioned \
        --dst-crs EPSG:32617
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

import pipeline  # noqa: F401  -- side-effecting import applies the pysheds/NumPy shim

# D8 direction map, pysheds default ordering: (N, NE, E, SE, S, SW, W, NW).
DIRMAP: tuple[int, ...] = (64, 128, 1, 2, 4, 8, 16, 32)

# Sentinel for nodata in the float32 working/output rasters. Reprojecting a
# geographic tile into a metric grid leaves nodata wedges in the corners; this
# marks them. Chosen to match SRTM's int16 void value so it reads naturally.
NODATA: float = -32768.0


@dataclass
class ConditionedDEM:
    """Conditioning outputs + the georeferencing needed to write/tile them.

    Arrays are all the same shape ``(height, width)`` on a single metric grid.
    """

    bed: np.ndarray  # float32, true reprojected elevation (physical bed); NODATA in corners
    filled: np.ndarray  # float32, sink-filled + flats-resolved hydrologic surface
    flowdir: np.ndarray  # D8 direction codes (see DIRMAP); 0 where nodata
    flow_accum: np.ndarray  # float32, upstream contributing-cell count
    crs: str  # e.g. "EPSG:32617"
    transform: tuple[float, float, float, float, float, float]  # affine (a,b,c,d,e,f)
    dx: float  # metric cell size (metres); grid is square within DX_TOL
    bounds: tuple[float, float, float, float]  # (left, bottom, right, top) in CRS units
    nodata: float
    dirmap: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return self.bed.shape  # type: ignore[return-value]


# Largest tolerated relative difference between x- and y-pixel size before we
# refuse to call the grid "square" (the solver assumes a single dx).
DX_TOL = 1e-3


def reproject_to_metric(
    src_path: str | Path,
    dst_crs: str = "EPSG:32617",
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, rasterio.Affine, str, float]:
    """Reproject a DEM to a metric CRS with square pixels.

    Returns ``(elevation_f32, transform, crs_string, dx_metres)``. Nodata cells
    (including the reprojection corner wedges) are set to ``NODATA``.
    """
    dst = CRS.from_string(dst_crs)
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst, src.width, src.height, *src.bounds
        )
        out = np.full((height, width), NODATA, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=dst,
            dst_nodata=NODATA,
            resampling=resampling,
        )

    dx, dy = abs(transform.a), abs(transform.e)
    if abs(dx - dy) / dx > DX_TOL:
        raise ValueError(f"non-square pixels after reprojection: dx={dx:.4f} dy={dy:.4f}")
    return out, transform, dst.to_string(), float(dx)


def condition_array(
    elevation: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    dx: float,
    dirmap: tuple[int, ...] = DIRMAP,
) -> ConditionedDEM:
    """Run the pysheds conditioning chain on an already-metric elevation grid."""
    bed = np.asarray(elevation, dtype=np.float32)
    vf = ViewFinder(
        shape=bed.shape,
        affine=transform,
        crs=pyproj.Proj(crs),
        nodata=np.float32(NODATA),
    )
    grid = Grid(viewfinder=vf)
    dem = Raster(bed, viewfinder=vf)

    filled = grid.fill_depressions(dem)
    filled = grid.resolve_flats(filled)
    fdir = grid.flowdir(filled, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    left, bottom, right, top = (
        transform.c,
        transform.f + transform.e * bed.shape[0],
        transform.c + transform.a * bed.shape[1],
        transform.f,
    )
    return ConditionedDEM(
        bed=bed,
        filled=np.asarray(filled, dtype=np.float32),
        flowdir=np.asarray(fdir),
        flow_accum=np.asarray(acc, dtype=np.float32),
        crs=crs,
        transform=tuple(transform)[:6],
        dx=float(dx),
        bounds=(float(left), float(bottom), float(right), float(top)),
        nodata=NODATA,
        dirmap=dirmap,
    )


def _write_geotiff(path: Path, data: np.ndarray, transform: rasterio.Affine, crs: str) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=NODATA if np.issubdtype(data.dtype, np.floating) else None,
        compress="deflate",
        predictor=2,
    ) as dst:
        dst.write(data, 1)


def write_outputs(cond: ConditionedDEM, out_dir: str | Path) -> dict[str, str]:
    """Write the conditioned rasters + a metadata JSON. Returns the file map."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    transform = rasterio.Affine(*cond.transform)

    layers = {
        "bed_elevation": cond.bed,
        "filled_elevation": cond.filled,
        "flow_direction": cond.flowdir,
        "flow_accumulation": cond.flow_accum,
    }
    paths = {}
    for name, arr in layers.items():
        p = out / f"{name}.tif"
        _write_geotiff(p, arr, transform, cond.crs)
        paths[name] = str(p)

    valid = cond.bed[cond.bed != cond.nodata]
    meta = {
        "crs": cond.crs,
        "transform": list(cond.transform),
        "dx_m": cond.dx,
        "shape": list(cond.shape),
        "bounds": list(cond.bounds),
        "nodata": cond.nodata,
        "dirmap": list(cond.dirmap),
        "elevation_m": {
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "valid_cells": int(valid.size),
            "total_cells": int(cond.bed.size),
        },
        "flow_accumulation_max_cells": float(cond.flow_accum.max()),
        "layers": {k: Path(v).name for k, v in paths.items()},
    }
    meta_path = out / "condition.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    paths["metadata"] = str(meta_path)
    return paths


def condition_dem(
    src_path: str | Path,
    out_dir: str | Path,
    dst_crs: str = "EPSG:32617",
) -> ConditionedDEM:
    """Full M0 conditioning: reproject -> fill -> flowdir -> accumulation -> write."""
    elevation, transform, crs, dx = reproject_to_metric(src_path, dst_crs)
    cond = condition_array(elevation, transform, crs, dx)
    write_outputs(cond, out_dir)
    return cond


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Condition a DEM for the river-basin pipeline.")
    p.add_argument("--src", required=True, help="raw DEM path (e.g. data/dem/raw/N35W083.hgt)")
    p.add_argument("--out", required=True, help="output directory for conditioned rasters")
    p.add_argument("--dst-crs", default="EPSG:32617", help="metric target CRS (default UTM 17N)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cond = condition_dem(args.src, args.out, args.dst_crs)
    h, w = cond.shape
    valid = cond.bed[cond.bed != cond.nodata]
    print(f"conditioned {args.src}")
    print(f"  crs        : {cond.crs}")
    print(f"  grid       : {w} x {h} @ dx={cond.dx:.2f} m")
    print(f"  elevation  : {valid.min():.0f}..{valid.max():.0f} m (mean {valid.mean():.0f})")
    print(f"  max accum  : {cond.flow_accum.max():.0f} cells")
    print(f"  written to : {args.out}")


if __name__ == "__main__":
    main()
