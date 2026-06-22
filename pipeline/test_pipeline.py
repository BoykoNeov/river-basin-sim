"""Pipeline tests (M0): DEM conditioning + tiling on tiny synthetic data.

These exercise the real pysheds chain (so they also guard the NumPy-2.x
``np.in1d`` shim in ``pipeline/_compat``) and the ``.r32`` tile round-trip. They
need the ``geo`` extra (``uv sync --extra geo``); without it they skip cleanly so
a minimal ``uv run pytest`` stays green. No GPU required.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# Skip the whole module if the offline geo stack is not installed.
pytest.importorskip("pysheds")
rasterio = pytest.importorskip("rasterio")

from pipeline.condition import (  # noqa: E402  (after importorskip by design)
    DIRMAP,
    NODATA,
    condition_array,
    condition_dem,
)
from pipeline.tile import auto_window, read_conditioned, tile_dem  # noqa: E402


def _synthetic_dem(n: int = 48, pit: bool = True) -> np.ndarray:
    """A radial bowl (flow converges inward) with an optional artificial pit."""
    yy, xx = np.mgrid[0:n, 0:n]
    dem = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2).astype(np.float32)
    if pit:
        dem[n // 2, n // 2] = -50.0
    return dem


def _write_geographic_dem(path, n: int = 48) -> None:
    """Write a tiny EPSG:4326 GeoTIFF so condition_dem can reproject it."""
    dem = _synthetic_dem(n)
    # A small patch inside UTM zone 17N (~82.5 W, 35.5 N) so reprojection to
    # EPSG:32617 is well-posed; 0.01 deg/cell. Exact extent is unimportant.
    transform = rasterio.Affine(0.01, 0, -82.5, 0, -0.01, 35.5)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(dem, 1)


def test_condition_array_fills_pit_and_routes_flow():
    n = 48
    dem = _synthetic_dem(n)
    transform = rasterio.Affine(30.0, 0, 0, 0, -30.0, 0)  # metric, 30 m square
    cond = condition_array(dem, transform, "EPSG:32617", dx=30.0)

    # The artificial pit (-50) must be filled away.
    assert cond.filled.min() > -50.0
    # Flow directions are valid D8 codes (plus 0 for nodata / -1,-2 pysheds flags).
    valid_codes = set(DIRMAP) | {0, -1, -2}
    assert set(np.unique(cond.flowdir).tolist()).issubset(valid_codes)
    # A bowl concentrates flow: the outlet accumulates many cells.
    assert cond.flow_accum.max() >= n  # at least a row's worth converges
    assert cond.dx == 30.0
    assert cond.bed.shape == (n, n)


def test_condition_dem_end_to_end(tmp_path):
    src = tmp_path / "dem.tif"
    out = tmp_path / "conditioned"
    _write_geographic_dem(src)

    cond = condition_dem(src, out, dst_crs="EPSG:32617")

    # Outputs + metadata exist.
    for name in ("bed_elevation", "filled_elevation", "flow_direction", "flow_accumulation"):
        assert (out / f"{name}.tif").exists()
    meta = json.loads((out / "condition.json").read_text())
    assert meta["crs"] == "EPSG:32617"
    assert meta["dx_m"] > 0  # reprojected to metres
    # Reprojection introduces square pixels; condition_array enforced the tolerance.
    assert cond.dx > 0


def test_tile_round_trip(tmp_path):
    # Condition a synthetic DEM, then cut a single small tile and round-trip it.
    src = tmp_path / "dem.tif"
    cond_dir = tmp_path / "conditioned"
    tiles_dir = tmp_path / "tiles"
    _write_geographic_dem(src, n=64)
    condition_dem(src, cond_dir, dst_crs="EPSG:32617")

    manifest = tile_dem(cond_dir, tiles_dir, size=32, single=True)
    assert len(manifest["tiles"]) == 1
    entry = manifest["tiles"][0]
    assert entry["width"] == 32 and entry["height"] == 32

    # The .r32 is raw little-endian float32, row-major -> reload and compare.
    raw = np.fromfile(tiles_dir / entry["file"], dtype="<f4")
    assert raw.size == 32 * 32
    img = raw.reshape(entry["height"], entry["width"])
    assert np.isclose(float(img.min()), entry["h_min"], atol=1e-3)
    assert np.isclose(float(img.max()), entry["h_max"], atol=1e-3)
    # Manifest carries the contract metadata the viewer/solver rely on.
    assert manifest["dtype"] == "<f4"
    assert manifest["dx_m"] > 0


def test_auto_window_prefers_clean_interior():
    # A grid whose only nodata is a corner; auto_window must avoid it when clean
    # windows exist, and the chosen window must be nodata-free.
    bed = np.full((80, 80), 100.0, dtype=np.float32)
    bed[:16, :16] = NODATA  # nodata corner
    acc = np.ones((80, 80), dtype=np.float32)
    acc[40, 40] = 1e6  # a strong "river" cell in the interior
    r0, c0 = auto_window(bed, acc, size=32)
    window = bed[r0 : r0 + 32, c0 : c0 + 32]
    assert not np.any(window == NODATA)


def test_read_conditioned_matches_metadata(tmp_path):
    src = tmp_path / "dem.tif"
    cond_dir = tmp_path / "conditioned"
    _write_geographic_dem(src, n=48)
    condition_dem(src, cond_dir, dst_crs="EPSG:32617")

    bed, acc, meta, transform = read_conditioned(cond_dir)
    assert bed.shape == tuple(meta["shape"])
    assert acc.shape == bed.shape
    assert abs(transform.a - meta["dx_m"]) < 1e-6
