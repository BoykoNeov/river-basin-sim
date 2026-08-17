"""Channel-geometry tests: does the derived river actually convey?

Deliberately **not** behind the ``geo`` extra's ``importorskip`` (unlike
``test_pipeline.py``): these gate a defect that no other test in the repo can see,
so they must run in a bare ``uv run pytest``. Everything here is pure numpy --
``pipeline.channels`` only imports rasterio inside :func:`channel_fields`.

The defect (``docs/plans/real-dem-reach.md`` §2): a D8 flow path steps diagonally
about half the time, the solver's channel faces are N/S/E/W only, and face width is
``min(w_L, w_R)`` -- so an unfixed network is a chain of sealed pools that conserves
mass perfectly while conveying nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.channels import (
    DEFAULT_DIRMAP,
    DIRMAP_OFFSETS,
    components,
    connectivity_report,
    d8_offsets,
    hydraulic_geometry,
    isolated_cells,
    rook_connect,
)

SE = 2  # DEFAULT_DIRMAP's south-east code; (dr, dc) = (1, 1)


def _diagonal_river(n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """A staircase river running SE down the leading diagonal.

    Drainage area grows downstream (1, 2, ... km²) as D8 accumulation does; every
    off-path cell is hillslope. This is the pathological-but-ordinary case: a real
    D8 path takes this step 48 % of the time.
    """
    area = np.zeros((n, n), dtype=np.float64)
    fdir = np.zeros((n, n), dtype=np.int64)
    for i in range(n - 1):
        area[i, i] = i + 1.0
        fdir[i, i] = SE
    area[n - 1, n - 1] = float(n)
    return area, fdir


# --- the component finder itself, before anything is measured with it ---------


def test_component_finder_separates_4_and_8_connectivity():
    """A bare diagonal is N components under rook moves and 1 under king moves."""
    m = np.eye(6, dtype=bool)
    assert components(m, diagonal=False) == (6, 1)
    assert components(m, diagonal=True) == (1, 6)


def test_component_finder_handles_the_empty_and_full_cases():
    assert components(np.zeros((4, 4), bool)) == (0, 0)
    assert components(np.ones((4, 4), bool), diagonal=False) == (1, 16)


def test_d8_offsets_follow_the_recorded_dirmap_order():
    off = d8_offsets(DEFAULT_DIRMAP)
    assert off[64] == (-1, 0)  # N -- rows increase southward
    assert off[SE] == (1, 1)
    assert tuple(off.values()) == DIRMAP_OFFSETS
    with pytest.raises(ValueError):
        d8_offsets((1, 2, 4))


# --- the defect -------------------------------------------------------------


def test_a_raw_d8_network_does_not_convey():
    """Every cell of a diagonal river is a sealed pool on a 4-connected grid."""
    area, _ = _diagonal_river()
    w, _, _ = hydraulic_geometry(area, dx=1000.0, min_area_km2=1.0)
    rep = connectivity_report(w)

    assert rep["isolated"] == rep["channel_cells"]  # every single one
    assert rep["components_4"] == rep["channel_cells"]
    assert rep["components_8"] == 1  # the same mask is one coherent river
    assert rep["largest_4"] == 1


# --- the fix ----------------------------------------------------------------


def test_rook_connect_leaves_no_cell_without_a_4_connected_neighbour():
    """The gate, in its binary form -- no tolerance to calibrate by its answer."""
    area, fdir = _diagonal_river()
    fixed, inserted = rook_connect(area, fdir, min_area_km2=1.0)
    w, _, _ = hydraulic_geometry(fixed, dx=1000.0, min_area_km2=1.0)

    assert inserted > 0
    assert isolated_cells(w > 0.0) == 0


def test_rook_connect_makes_the_4_connected_structure_match_the_8_connected_one():
    area, fdir = _diagonal_river()
    fixed, _ = rook_connect(area, fdir, min_area_km2=1.0)
    rep = connectivity_report(hydraulic_geometry(fixed, dx=1000.0, min_area_km2=1.0)[0])

    assert rep["components_4"] == rep["components_8"] == 1
    assert rep["largest_4"] == rep["largest_8"] == rep["channel_cells"]


def test_the_inserted_cell_inherits_the_through_river_not_its_own_area():
    """Insertion alone leaves a pinhole; inheriting the through-width is the fix.

    A corner cell sits *beside* the river, so its own drainage area is negligible --
    here exactly zero, which is the honest limit of the ~450x median shortfall
    measured on the real DEM. Sized by its own area it carries no channel at all, and
    ``w_face = min(w_L, w_R)`` puts the wall straight back.
    """
    area, fdir = _diagonal_river()
    fixed, _ = rook_connect(area, fdir, min_area_km2=1.0)

    # The step (3,3) -> (4,4) routes through (4,3): the vertical-first corner.
    assert area[4, 3] == 0.0  # its own drainage area: hillslope
    assert fixed[4, 3] == area[3, 3]  # it inherits the river making the step

    w, d, _ = hydraulic_geometry(fixed, dx=1000.0, min_area_km2=1.0)
    assert w[4, 3] == pytest.approx(w[3, 3])
    assert d[4, 3] == pytest.approx(d[3, 3])
    # min(w_L, w_R) across the inserted face is the full river, not a pinhole.
    assert min(w[3, 3], w[4, 3]) == pytest.approx(w[3, 3])


def test_a_river_leaving_the_window_is_not_counted_as_a_broken_network():
    """A field is a window cut from a larger raster; its border cells continue outside.

    Measured on the real DEM's M0 tile: after the fix exactly one cell is isolated and
    it sits on row 0. Counting that as a defect would make the gate unpassable for any
    windowed domain, which is every domain.
    """
    m = np.zeros((8, 8), dtype=bool)
    m[0, 4] = True  # a river cell on the border, nothing beside it
    m[3, 1:4] = True  # a genuine interior run, properly connected
    assert isolated_cells(m) == 1
    assert isolated_cells(m, interior_only=True) == 0

    m[5, 5] = True  # a genuine interior orphan
    assert isolated_cells(m, interior_only=True) == 1


def test_rook_connect_is_idempotent():
    area, fdir = _diagonal_river()
    once, n1 = rook_connect(area, fdir, min_area_km2=1.0)
    twice, n2 = rook_connect(once, fdir, min_area_km2=1.0)

    assert n2 == 0
    np.testing.assert_array_equal(once, twice)


def test_rook_connect_leaves_a_cardinal_network_untouched():
    """A river that only steps N/S/E/W already has faces -- nothing to insert."""
    area = np.zeros((8, 8))
    fdir = np.zeros((8, 8), dtype=np.int64)
    area[4, :] = np.arange(1, 9)
    fdir[4, :] = 1  # E
    fixed, inserted = rook_connect(area, fdir, min_area_km2=1.0)

    assert inserted == 0
    np.testing.assert_array_equal(fixed, area)


def test_rook_connect_does_not_step_outside_the_domain():
    """A diagonal step off the edge is dropped, not wrapped or clipped into a lie."""
    area = np.zeros((4, 4))
    fdir = np.zeros((4, 4), dtype=np.int64)
    area[3, 3] = 5.0
    fdir[3, 3] = SE  # points off the south-east corner
    fixed, inserted = rook_connect(area, fdir, min_area_km2=1.0)

    assert inserted == 0
    np.testing.assert_array_equal(fixed, area)


# --- the clip resolution ----------------------------------------------------


def test_the_width_clip_is_taken_at_the_resolution_passed_in():
    """Clipping at the tile dx while the run steps at coarsen*dx understates the river.

    ``solver.io.coarsen`` aggregates width by block *max*, so a width clipped away
    before coarsening cannot be recovered -- the clip has to be taken against the
    resolution the run will actually use.
    """
    area = np.array([[1.0, 25.0, 400.0, 2500.0]])  # w = 8, 40, 160, 400 m

    w_fine, _, n_fine = hydraulic_geometry(area, dx=50.0, min_area_km2=1.0)
    w_coarse, _, n_coarse = hydraulic_geometry(area, dx=200.0, min_area_km2=1.0)

    assert n_fine == 2 and n_coarse == 1  # coarser cells clip fewer rivers
    assert n_coarse < n_fine
    assert w_fine[0, 2] == 50.0 and w_coarse[0, 2] == pytest.approx(160.0)
    # The understatement the defect produced: 3.2x on this cell alone.
    assert w_coarse[0, 2] / w_fine[0, 2] == pytest.approx(3.2)
    # Both still refuse a channel wider than its own cell.
    assert w_fine.max() <= 50.0 and w_coarse.max() <= 200.0


def test_a_clipped_width_survives_the_float32_cast_as_still_within_the_cell():
    """``validate_geometry`` rejects on a strict ``w > dx``, and float32 can round up.

    The real DEM's cell size at ``coarsen = 4`` is 112.58551545578392 m, whose nearest
    float32 is 112.58551788 -- *larger*. Clipping to ``dx`` in float64 and then casting
    would hand the solver a width it refuses, from a field that is correct by
    construction.
    """
    dx = 28.14637886394598 * 4
    w, _, clipped = hydraulic_geometry(np.array([[5000.0]]), dx=dx, min_area_km2=1.0)

    assert clipped == 1
    assert w.dtype == np.float32
    # The comparison the solver actually makes, at both widths of the promotion rule.
    assert not bool(w[0, 0] > np.float32(dx))
    assert float(w[0, 0]) <= dx
    # Still the cell width to any tolerance that matters -- this is not a haircut.
    assert float(w[0, 0]) == pytest.approx(dx, rel=1e-6)
