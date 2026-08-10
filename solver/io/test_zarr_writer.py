"""The canonical store's optional morphology field (M7 build step 7, §7.2)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from solver.core.grid import Grid
from solver.io.zarr_writer import ZarrWriter


def _grid() -> Grid:
    return Grid(ny=3, nx=4, dx=10.0)


def _frame(g: Grid, fill: float) -> np.ndarray:
    return np.full(g.shape, fill, dtype=np.float32)


def test_a_store_without_morphology_has_no_bed_change_array(tmp_path):
    """The M6 store shape, unchanged: no array, no attribute, nothing to ignore.

    The invariant every milestone since M4 has held -- a run that does not use the
    new feature must produce what it produced before. A ``bed_change`` full of zeros
    would satisfy a reader and still be a new array in every existing store.
    """
    g = _grid()
    w = ZarrWriter(tmp_path / "r.zarr", g, 2, {"dx": g.dx})
    w.write_bed(_frame(g, 100.0))
    for i in range(2):
        w.append(float(i), _frame(g, 0.1), _frame(g, 0.0), _frame(g, 0.0))
    w.finalize()

    ds = xr.open_zarr(tmp_path / "r.zarr", consolidated=False)
    assert "bed_change" not in ds
    assert set(ds.data_vars) == {"depth", "u", "v", "bed"}


def test_bed_change_rides_the_time_axis_and_opens_with_xarray(tmp_path):
    """``(T, Y, X)`` float32 beside ``depth``, same chunking, same dimension names.

    The dimension names are not decoration: without them ``xr.open_zarr`` does not
    surface the array at all, and :mod:`solver.io.viewer_export` -- which reads the
    store through xarray to decide whether the run morphed -- would quietly see a
    store with no morphology rather than fail.
    """
    g = _grid()
    w = ZarrWriter(tmp_path / "r.zarr", g, 3, {"dx": g.dx}, bed_change=True)
    w.write_bed(_frame(g, 100.0))
    for i in range(3):
        w.append(float(i), _frame(g, 0.1), _frame(g, 0.0), _frame(g, 0.0), _frame(g, -0.01 * i))
    w.finalize()

    ds = xr.open_zarr(tmp_path / "r.zarr", consolidated=False)
    assert ds["bed_change"].dims == ("time", "y", "x")
    assert ds["bed_change"].shape == (3, *g.shape)
    assert ds["bed_change"].dtype == np.float32
    assert ds["bed_change"].encoding["chunks"] == (1, *g.shape)
    assert float(ds["bed_change"].isel(time=2).min()) == pytest.approx(-0.02)


def test_a_missing_or_unexpected_bed_change_frame_is_an_error_both_ways(tmp_path):
    """Zero is a legal bed change, so an omitted frame cannot be detected later.

    A preallocated frame reads as "the bed did not move here", which is a statement
    about the physics and not a visible hole -- the same class of quiet lie as
    keeping depth non-negative with ``max(h, 0)``. So the mismatch is refused at the
    call, in **both** directions: a store that carries the field always gets one, and
    a store that does not never gets one.
    """
    g = _grid()
    morphing = ZarrWriter(tmp_path / "a.zarr", g, 1, {}, bed_change=True)
    with pytest.raises(ValueError, match="bed did not move"):
        morphing.append(0.0, _frame(g, 0.1), _frame(g, 0.0), _frame(g, 0.0))

    still = ZarrWriter(tmp_path / "b.zarr", g, 1, {})
    with pytest.raises(ValueError, match="exactly when the store carries it"):
        still.append(0.0, _frame(g, 0.1), _frame(g, 0.0), _frame(g, 0.0), _frame(g, 0.0))
