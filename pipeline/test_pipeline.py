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


# --- M6: sub-grid channel geometry -------------------------------------------


def test_hydraulic_geometry_scales_with_area_and_clips_to_the_cell():
    """Width grows with drainage area, stops at dx, and vanishes below threshold."""
    from pipeline.channels import hydraulic_geometry

    area = np.array([[0.1, 1.0, 10.0, 1000.0]])  # km^2
    w, d, clipped = hydraulic_geometry(area, dx=50.0, min_area_km2=1.0)

    assert w[0, 0] == 0.0 and d[0, 0] == 0.0  # hillslope: no channel
    assert w[0, 1] == pytest.approx(8.0)  # 8 * 1^0.5
    assert w[0, 2] == pytest.approx(8.0 * 10**0.5, rel=1e-5)
    assert w[0, 3] == 50.0  # clipped to dx -- a channel must fit its cell
    assert clipped == 1
    assert d[0, 2] == pytest.approx(0.27 * 10**0.3, rel=1e-5)
    # Monotone in area, and "no channel" is zero in both fields.
    assert np.all(np.diff(w[0]) >= 0)
    assert np.all((w[0] > 0) == (d[0] > 0))


def test_channel_fields_align_to_the_tile_mosaic(tmp_path):
    """The fields the solver loads must match the assembled domain, cell for cell."""
    from pipeline.channels import channel_fields

    src = tmp_path / "dem.tif"
    cond_dir = tmp_path / "conditioned"
    tiles_dir = tmp_path / "tiles"
    fields_dir = tmp_path / "fields"
    _write_geographic_dem(src, n=64)
    condition_dem(src, cond_dir, dst_crs="EPSG:32617")
    manifest = tile_dem(cond_dir, tiles_dir, size=32)  # a full multi-tile mosaic

    record = channel_fields(cond_dir, tiles_dir, fields_dir, min_area_km2=0.0)

    ny = max(t["row"] + t["height"] for t in manifest["tiles"])
    nx = max(t["col"] + t["width"] for t in manifest["tiles"])
    assert record["shape"] == [ny, nx]
    for name in ("channel_width", "channel_depth"):
        raw = np.fromfile(fields_dir / f"{name}.r32", dtype="<f4")
        assert raw.size == ny * nx
    # The coefficients travel with the fields -- they are calibration, not constants.
    assert "coefficients" in json.loads((fields_dir / "channels.json").read_text())


def test_channel_fields_threads_coarsen_and_connectivity_through(tmp_path):
    """The run resolution and the connectivity fix reach the fields and the record.

    Behaviour is gated in ``test_channels.py`` on inputs it controls exactly; this is
    the plumbing check -- that ``coarsen`` is not silently the tile ``dx`` again, and
    that the connectivity pass ran and said what it did.
    """
    from pipeline.channels import channel_fields

    src = tmp_path / "dem.tif"
    cond_dir = tmp_path / "conditioned"
    tiles_dir = tmp_path / "tiles"
    _write_geographic_dem(src, n=64)
    condition_dem(src, cond_dir, dst_crs="EPSG:32617")
    tile_dem(cond_dir, tiles_dir, size=32)

    fine = channel_fields(cond_dir, tiles_dir, tmp_path / "f1", coarsen=1, min_area_km2=0.0)
    coarse = channel_fields(cond_dir, tiles_dir, tmp_path / "f4", coarsen=4, min_area_km2=0.0)

    assert fine["run_dx_m"] == pytest.approx(fine["dx_m"])
    assert coarse["run_dx_m"] == pytest.approx(fine["dx_m"] * 4)
    # A coarser run cell cannot clip more rivers than a finer one.
    assert coarse["width_clipped_to_run_dx"] <= fine["width_clipped_to_run_dx"]
    assert coarse["coarsen"] == 4

    # The connectivity pass ran, reported, and can be turned off.
    assert fine["rook_connected"] is True
    assert set(fine["connectivity"]) >= {"isolated", "components_4", "components_8"}
    raw = channel_fields(cond_dir, tiles_dir, tmp_path / "f0", connect=False, min_area_km2=0.0)
    assert raw["rook_connected"] is False
    assert raw["cells_inserted_for_connectivity"] == 0

    with pytest.raises(ValueError):
        channel_fields(cond_dir, tiles_dir, tmp_path / "bad", coarsen=0)
