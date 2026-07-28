"""Tile-mosaic assembly tests (M6 plan §3 -- tiling at scale).

The domain is now the tile *set*, so the gates are: a tiled array reassembles to
its source exactly, a run over a mosaic is bitwise-identical to the same run over
the equivalent single bed (the seam cannot perturb the solver), gaps are counted
rather than hidden, and a window clips without reading tiles it does not need.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from solver.io.mosaic import MosaicError, assemble_mosaic
from solver.run import Scenario, run_simulation


def _terrain(ny: int, nx: int) -> np.ndarray:
    """A non-symmetric bed so a misplaced tile cannot pass by coincidence."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (100.0 + 0.05 * yy + 0.11 * xx + 0.3 * np.sin(0.4 * yy) * np.cos(0.3 * xx)).astype(
        np.float32
    )


def _write_tiles(tiles_dir, source: np.ndarray, size: int, *, dx: float = 30.0) -> dict:
    """Cut ``source`` into ``size``-square .r32 tiles + an M0-shaped tiles.json."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    ny, nx = source.shape
    entries = []
    for r0 in range(0, ny, size):
        for c0 in range(0, nx, size):
            block = np.ascontiguousarray(source[r0 : r0 + size, c0 : c0 + size], dtype="<f4")
            name = f"tile_{r0 // size:02d}_{c0 // size:02d}.r32"
            (tiles_dir / name).write_bytes(block.tobytes(order="C"))
            entries.append(
                {
                    "file": name,
                    "row": r0,
                    "col": c0,
                    "width": int(block.shape[1]),
                    "height": int(block.shape[0]),
                }
            )
    manifest = {"crs": "EPSG:32617", "dx_m": dx, "tile_size": size, "tiles": entries}
    (tiles_dir / "tiles.json").write_text(json.dumps(manifest))
    return manifest


def test_mosaic_reassembles_the_source_exactly(tmp_path):
    source = _terrain(64, 96)  # 2 x 3 tiles of 32
    _write_tiles(tmp_path / "tiles", source, 32)

    m = assemble_mosaic(tmp_path / "tiles")

    assert m.shape == (64, 96)
    assert m.tiles_used == m.tiles_total == 6
    assert m.gap_cells == 0
    assert m.dx == 30.0 and m.crs == "EPSG:32617"
    np.testing.assert_array_equal(m.bed, source)


def test_edge_tiles_may_be_clipped(tmp_path):
    source = _terrain(50, 40)  # 32-tiles -> a clipped last row/column
    _write_tiles(tmp_path / "tiles", source, 32)

    m = assemble_mosaic(tmp_path / "tiles")

    assert m.shape == (50, 40)
    np.testing.assert_array_equal(m.bed, source)


def test_first_selects_only_tile_zero(tmp_path):
    """The pre-M6 behaviour stays reachable -- and is now an explicit choice."""
    source = _terrain(64, 64)
    _write_tiles(tmp_path / "tiles", source, 32)

    m = assemble_mosaic(tmp_path / "tiles", select="first")

    assert m.shape == (32, 32)
    assert m.tiles_used == 1 and m.tiles_total == 4
    np.testing.assert_array_equal(m.bed, source[:32, :32])


def test_window_clips_a_reach_out_of_the_mosaic(tmp_path):
    source = _terrain(64, 64)
    _write_tiles(tmp_path / "tiles", source, 32)

    m = assemble_mosaic(tmp_path / "tiles", window=(20, 28, 39, 47))  # inclusive

    assert m.shape == (20, 20)
    assert m.origin == (20, 28)
    np.testing.assert_array_equal(m.bed, source[20:40, 28:48])
    # Only the tiles the window touches are read -- here all four, since the window
    # straddles both tile seams; a window inside one tile reads exactly one.
    assert m.tiles_used == 4
    inner = assemble_mosaic(tmp_path / "tiles", window=(2, 2, 9, 9))
    assert inner.tiles_used == 1
    np.testing.assert_array_equal(inner.bed, source[2:10, 2:10])


def test_gaps_are_counted_and_filled_not_hidden(tmp_path):
    """An L-shaped tile set leaves a hole; it is filled flat *and* reported."""
    source = _terrain(64, 64)
    _write_tiles(tmp_path / "tiles", source, 32)
    manifest = json.loads((tmp_path / "tiles" / "tiles.json").read_text())
    dropped = manifest["tiles"].pop()  # remove the bottom-right tile
    (tmp_path / "tiles" / "tiles.json").write_text(json.dumps(manifest))

    m = assemble_mosaic(tmp_path / "tiles")

    assert m.shape == (64, 64)
    assert m.gap_cells == 32 * 32
    assert m.tiles_used == 3 and m.tiles_total == 3
    hole = m.bed[32:, 32:]
    assert np.all(hole == m.fill_value)
    assert m.fill_value == pytest.approx(float(m.bed[:32, :].min()))
    assert "gap cells" in m.summary()
    assert dropped["file"] not in m.summary()


def test_bad_manifests_fail_loudly(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(MosaicError, match="no tile manifest"):
        assemble_mosaic(tmp_path / "empty")

    d = tmp_path / "notiles"
    d.mkdir()
    (d / "tiles.json").write_text(json.dumps({"tiles": []}))
    with pytest.raises(MosaicError, match="lists no tiles"):
        assemble_mosaic(d)

    source = _terrain(32, 32)
    _write_tiles(tmp_path / "tiles", source, 32)
    with pytest.raises(MosaicError, match="outside the mosaic"):
        assemble_mosaic(tmp_path / "tiles", window=(0, 0, 40, 40))
    with pytest.raises(MosaicError, match="must be one of"):
        assemble_mosaic(tmp_path / "tiles", select="every-other")


def test_run_over_a_mosaic_is_bitwise_identical_to_the_single_bed(tmp_path):
    """The assembly seam is I/O only: the solver must not be able to tell.

    Same terrain, once as a 2x2 tile set and once as one array -> the depth field
    must agree **bit for bit**, not approximately (HANDOFF §12 determinism).
    """
    source = _terrain(48, 48)
    _write_tiles(tmp_path / "tiles", source, 24)
    mosaic = assemble_mosaic(tmp_path / "tiles")
    np.testing.assert_array_equal(mosaic.bed, source)

    scn = Scenario(
        name="mosaic_equiv",
        dx=30.0,
        end_time=600.0,
        output_every=300.0,
        dt_max=10.0,
        rain_mm_hr=120.0,
        rain_duration=600.0,
    )
    a = run_simulation(scn, mosaic.bed, tmp_path / "a.zarr", device="cpu", verbose=False)
    b = run_simulation(scn, source, tmp_path / "b.zarr", device="cpu", verbose=False)

    da = xr.open_zarr(tmp_path / "a.zarr", consolidated=False)["depth"].values
    db = xr.open_zarr(tmp_path / "b.zarr", consolidated=False)["depth"].values
    np.testing.assert_array_equal(da, db)
    assert a.max_rel_error == b.max_rel_error
