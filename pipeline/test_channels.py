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
    component_labels,
    components,
    connectivity_report,
    d8_offsets,
    drainage_check,
    hydraulic_geometry,
    isolated_cells,
    isolation_cause,
    rook_connect,
    route_report,
    subgrid_cutoff_km2,
    trace_downstream,
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


# --- the cutoff: what counts as a river at this resolution -------------------


def test_the_subgrid_cutoff_is_the_area_whose_channel_is_exactly_one_cell_wide():
    """It inverts the width law, so it moves with the run resolution and the coefficients.

    198.1 km² at the real DEM's ``coarsen = 4`` cell and 12.4 km² at its native 28.15 m
    -- the same terrain, a different answer to "what is a river here", which is why the
    number has to be recorded rather than assumed.
    """
    dx = 28.14637886394598
    assert subgrid_cutoff_km2(dx * 4) == pytest.approx(198.1, abs=0.05)
    assert subgrid_cutoff_km2(dx) == pytest.approx(12.4, abs=0.05)

    cut = subgrid_cutoff_km2(200.0)
    at_cut, _, _ = hydraulic_geometry(np.array([[cut, cut * 0.5]]), dx=200.0, min_area_km2=1.0)
    assert float(at_cut[0, 0]) == pytest.approx(200.0, rel=1e-6)  # exactly one cell
    assert float(at_cut[0, 1]) < 200.0


def test_the_cutoff_drops_exactly_the_rivers_that_would_have_been_clipped():
    """Above the cutoff the model is not sub-grid, it is "the river is one cell across".

    The cells the cutoff removes are precisely the ones the clip was flattening, and
    every cell below it is untouched -- so this changes what is carried, not how it is
    sized. Measured on the real DEM's M0 tile at ``coarsen = 4``: 2740 clipped cells
    become 0, and 2740 cells leave the channel mask.
    """
    area = np.array([[1.0, 25.0, 400.0, 2500.0]])  # w = 8, 40, 160, 400 m
    cut = subgrid_cutoff_km2(200.0)

    plain, d_plain, n_plain = hydraulic_geometry(area, dx=200.0, min_area_km2=1.0)
    cutd, d_cut, n_cut = hydraulic_geometry(area, dx=200.0, min_area_km2=1.0, max_area_km2=cut)

    assert n_plain == 1 and n_cut == 0  # nothing is clipped once the big river is gone
    assert float(cutd.max()) <= 200.0
    assert cutd[0, 3] == 0.0 and d_cut[0, 3] == 0.0  # dropped, in both fields
    np.testing.assert_array_equal(cutd[0, :3], plain[0, :3])  # and nothing else moved
    np.testing.assert_array_equal(d_cut[0, :3], d_plain[0, :3])


def test_no_cutoff_is_the_default_and_leaves_the_field_bit_for_bit():
    """The cutoff is a modelling choice; every figure recorded before it assumed none."""
    area, fdir = _diagonal_river()
    fixed, _ = rook_connect(area, fdir, min_area_km2=1.0)
    a = hydraulic_geometry(fixed, dx=50.0, min_area_km2=1.0)
    b = hydraulic_geometry(fixed, dx=50.0, min_area_km2=1.0, max_area_km2=None)

    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_the_cutoff_cannot_orphan_a_cell_inserted_for_connectivity():
    """It is applied *after* :func:`rook_connect`, and that ordering is load-bearing.

    An inserted corner inherits ``min(area_upstream, area_downstream)`` = the upstream
    end's area, so whenever the corner's *own* area is also below the cutoff it survives
    exactly when the step it serves does. The exception is a corner that happens to sit
    on a bigger river of its own — it takes ``max(own, through)``, so it can be dropped
    while both ends of the diagonal step stay, and the wall comes back. It is a
    hillslope cell by construction, so this is rare rather than impossible, which is
    why the per-run 4-vs-8 component check is what actually verifies it.

    **It is rare and it is not impossible, and the full mosaic proves both halves.** On
    the single M0 tile the real derivation read 69 vs 69 with 0 isolated cells; on the
    whole 3991x3283 mosaic it reads 458 vs 456 with **9** isolated of 298 147 channel
    cells, against 163 vs 163 and 0 isolated for the same network with the cutoff off.
    So the ordering is still right and the exception above does fire -- which is why the
    cause is now reported as a measured comparison (:func:`isolation_cause`) instead of
    being read as the D8 defect.
    """
    area, fdir = _diagonal_river(n=16)
    fixed, inserted = rook_connect(area, fdir, min_area_km2=1.0)
    assert inserted > 0

    w, _, _ = hydraulic_geometry(fixed, dx=1000.0, min_area_km2=1.0, max_area_km2=8.5)
    mask = w > 0
    assert isolated_cells(mask, interior_only=True) == 0
    rep = connectivity_report(w)
    assert rep["components_4"] == rep["components_8"]


def test_a_shattered_network_is_blamed_on_the_cutoff_only_when_the_uncut_one_is_clean():
    """The two causes need different words, and the difference must be measured.

    A broken 4-connectivity gate means "this river has a wall across it and no other
    gate can see it" when it comes from the D8 defect, and "the trunk these tributaries
    hung from is floodplain by choice" when it comes from the cutoff. Reporting the
    second in the first's words sends the reader hunting for a bug that is not there.
    """
    clean = {"isolated_interior": 0, "components_4": 3, "components_8": 3}
    broken = {"isolated_interior": 4, "components_4": 9, "components_8": 6}

    # Gate satisfied: nothing to attribute, cutoff or not.
    assert isolation_cause(clean, None) == "clean"
    assert isolation_cause(clean, broken) == "clean"

    # No cutoff was applied, so there is no comparison and the D8 defect is the cause.
    assert isolation_cause(broken, None) == "d8"

    # The same network without the cutoff is clean -> the cutoff did it.
    assert isolation_cause(broken, clean) == "cutoff"

    # It was already broken before the cutoff -> the cutoff is not the story.
    assert isolation_cause(broken, broken) == "d8"


def test_the_cause_of_a_shattered_network_is_read_off_the_real_geometry():
    """End to end on the fixture, not on hand-written dicts.

    This builds the exception the test above names: a corner that sits on a bigger river
    of its own takes ``max(own, through)``, so a cutoff can drop it while both ends of
    the diagonal step survive -- and the wall it was inserted to remove comes back. The
    staircase is clean before the cutoff, so the attribution must name the cutoff and not
    the connectivity fix that ran first.

    Note the discriminating detail: merely *cutting* a network in two leaves both halves
    4-connected and the gate reads clean. What fires the gate is a step left bridged only
    diagonally, which is the D8 defect's own signature arriving by a different route.
    """
    area, fdir = _diagonal_river(n=16)
    # Give one corner cell a river of its own, larger than the cutoff will allow.
    area[8, 7] = 500.0
    fixed, inserted = rook_connect(area, fdir, min_area_km2=1.0)
    assert inserted > 0
    assert fixed[8, 7] == 500.0  # max(own, through) kept its own, bigger area

    uncut, _, _ = hydraulic_geometry(fixed, dx=1000.0, min_area_km2=1.0)
    uncut_rep = connectivity_report(uncut)
    assert isolation_cause(uncut_rep, None) == "clean"

    cut, _, _ = hydraulic_geometry(fixed, dx=1000.0, min_area_km2=1.0, max_area_km2=100.0)
    cut_rep = connectivity_report(cut)
    assert cut[8, 7] == 0.0  # the corner is gone, both ends of its step remain
    assert cut_rep["components_4"] > cut_rep["components_8"]
    assert isolation_cause(cut_rep, uncut_rep) == "cutoff"


# --- which piece is this cell in, and where does that piece drain? -----------


def _two_rivers() -> tuple[np.ndarray, np.ndarray]:
    """Two channels: one leaves the window, one dead-ends at an unresolved flat.

    The sealed one is shaped like the real DEM's second-largest piece: it *reaches* the
    window edge (its tributary starts on row 0) while draining to an interior cell the
    conditioning left with no flow direction. A bounding-box touch says nothing about
    where the water goes.
    """
    area = np.zeros((12, 12))
    fdir = np.zeros((12, 12), dtype=np.int64)

    # Sealed: a tributary down column 2 from the border, then east along row 9.
    for i in range(10):
        area[i, 2] = i + 1.0
        fdir[i, 2] = 4  # S
    for j in range(2, 10):
        area[9, j] = 10.0 + (j - 2)
        fdir[9, j] = 1  # E
    fdir[9, 9] = -2  # pysheds' "no direction here" -- the flat

    # Draining: east along row 3 and out of the window.
    for j in range(5, 12):
        area[3, j] = float(j - 4)
        fdir[3, j] = 1  # E
    return area, fdir


def test_component_labels_agree_with_the_counts_they_summarise():
    area, fdir = _two_rivers()
    w, _, _ = hydraulic_geometry(area, dx=1000.0, min_area_km2=1.0)
    labels, sizes = component_labels(w > 0)

    n, largest = components(w > 0)
    assert (n, largest) == (int(sizes.size), int(sizes.max())) == (2, 17)
    assert int((labels >= 0).sum()) == int(sizes.sum()) == int((w > 0).sum())
    assert labels[0, 2] == labels[9, 9] and labels[0, 2] != labels[3, 11]
    assert labels[0, 0] == -1  # off the mask


def test_a_flow_path_that_leaves_the_window_is_the_healthy_ending():
    _, fdir = _two_rivers()
    tr = trace_downstream(fdir, (3, 5))

    assert tr["reason"] == "left_domain"
    assert tr["steps"] == 7 and tr["end"] == [3, 12]


def test_a_flow_path_that_dead_ends_inside_the_domain_is_found():
    """95 cells of the real raster have no D8 code; the largest swallows 1262 km²."""
    _, fdir = _two_rivers()
    tr = trace_downstream(fdir, (0, 2))

    assert tr["reason"] == "no_direction"
    assert tr["code"] == -2
    assert tr["end"] == [9, 9]


def test_a_trace_stops_at_the_outlet_it_was_given_and_a_loop_terminates():
    _, fdir = _two_rivers()
    stop = np.zeros((12, 12), dtype=bool)
    stop[9, 5] = True
    tr = trace_downstream(fdir, (0, 2), stop=stop)
    assert tr["reason"] == "reached_stop" and tr["end"] == [9, 5]

    ring = np.zeros((4, 4), dtype=np.int64)
    ring[1, 1], ring[1, 2] = 1, 16  # E then W: two cells pointing at each other
    assert trace_downstream(ring, (1, 1))["reason"] == "loop"


def test_a_piece_that_reaches_the_window_edge_can_still_be_sealed():
    """The discriminating field is where the piece's *outlet* is, not its bounding box.

    On the real DEM's M0 tile all 27 pieces touch the edge, and the second largest of
    them (924 cells, 1262 km²) drains to a flat in the middle of the domain.
    """
    area, fdir = _two_rivers()
    w, _, _ = hydraulic_geometry(area, dx=1000.0, min_area_km2=1.0)
    rep = drainage_check(w, area, fdir)

    assert rep["components"] == 2
    assert rep["sealed_components"] == 1 and rep["sealed_cells"] == 17
    sealed = rep["sealed"][0]
    assert sealed["outlet"] == [9, 9] and sealed["outlet_on_edge"] is False
    assert sealed["reason"] == "no_direction" and sealed["code"] == -2
    # ...and that same piece does reach the border, which is why the bbox test fails.
    labels, _ = component_labels(w > 0)
    assert bool((labels[0, :] == labels[9, 9]).any())


def test_an_inlet_off_the_channel_is_reported():
    """``reach_basin`` injects at ``[4, 768]``, two cells off the meander, unnoticed
    for two milestones because rain wet the channel anyway."""
    area, fdir = _two_rivers()
    w, _, _ = hydraulic_geometry(area, dx=1000.0, min_area_km2=1.0)
    rep = route_report(w, fdir, [(5, 5)], [(3, 11)])

    assert rep["inlets"][0]["in_channel"] is False
    assert rep["inlets"][0]["component"] is None
    assert rep["same_component"] is None  # not a question that has an answer here
    assert any("not a channel cell" in m for m in rep["warnings"])


def test_an_inlet_and_an_outlet_in_different_pieces_are_reported():
    area, fdir = _two_rivers()
    w, _, _ = hydraulic_geometry(area, dx=1000.0, min_area_km2=1.0)

    apart = route_report(w, fdir, [(0, 2)], [(3, 11)])
    assert apart["same_component"] is False
    assert any("different pieces" in m for m in apart["warnings"])
    # ...and the inlet's own river dead-ends inside the domain, which is the reason.
    assert apart["inlets"][0]["route"]["dead_end_inside"] is True
    assert any("nowhere to go" in m for m in apart["warnings"])

    together = route_report(w, fdir, [(3, 6)], [(3, 11)])
    assert together["same_component"] is True
    assert together["warnings"] == []
    assert together["inlets"][0]["route"]["reason"] == "reached_stop"
    assert together["inlets"][0]["route"]["channel_steps"] == 6


def test_same_piece_is_a_question_about_a_pair_and_needs_both_cells():
    """Four inflow cells in one piece are not a route -- there is nothing to reach.

    "Check my inflow cells are in the river" is the obvious use with no outlet, and
    answering it with "inlet and outlet are in the same piece" would be a connection
    nobody established.
    """
    area, fdir = _two_rivers()
    w, _, _ = hydraulic_geometry(area, dx=1000.0, min_area_km2=1.0)

    inlets_only = route_report(w, fdir, [(3, 6), (3, 7), (3, 8)])
    assert [e["in_channel"] for e in inlets_only["inlets"]] == [True, True, True]
    assert {e["component"] for e in inlets_only["inlets"]} == {
        inlets_only["inlets"][0]["component"]
    }
    assert inlets_only["same_component"] is None
    assert inlets_only["outlets"] == []

    # An outlet off the channel leaves the pair unanswerable rather than answered.
    assert route_report(w, fdir, [(3, 6)], [(5, 5)])["same_component"] is None
