"""Coarsening tests (M6 plan §3) -- volume preservation, rules, index mapping.

Coarsening happens **once, before any water moves**, so the gates here are about
the aggregation itself: a block mean preserves volume exactly, a channel survives
its block, ``coarsen = 1`` is the identity (not merely equivalent), scenario cell
indices follow the resolution, and a coarse run still balances its mass.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from solver.core.massbalance import MASS_GATE
from solver.io.coarsen import (
    CoarsenError,
    block_reduce,
    check_indices,
    coarsen_scenario,
    coarsened_shape,
    crop_report,
)
from solver.io.config import Inflow, Structure
from solver.run import Scenario, run_simulation


def test_block_mean_preserves_volume_exactly():
    """A coarse cell's storage is its block's storage: sum(h)·dx² is invariant."""
    rng = np.random.default_rng(3)
    fine = rng.uniform(0.0, 4.0, (16, 24)).astype(np.float32)
    dx, k = 25.0, 4

    coarse = block_reduce(fine, k, "mean")

    assert coarse.shape == coarsened_shape(fine.shape, k) == (4, 6)
    v_fine = fine.astype(np.float64).sum() * dx * dx
    v_coarse = coarse.astype(np.float64).sum() * (dx * k) ** 2
    assert v_coarse == pytest.approx(v_fine, rel=1e-6)


def test_coarsen_of_one_is_the_identity():
    fine = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert block_reduce(fine, 1) is fine
    scn = Scenario(inflows=[Inflow(cell=(2, 3), hydrograph=[(0.0, 1.0)])])
    assert coarsen_scenario(scn, 1) is scn


def test_partial_blocks_are_cropped_and_reported():
    fine = np.ones((10, 7), dtype=np.float32)
    coarse = block_reduce(fine, 4, "mean")
    assert coarse.shape == (2, 1)
    note = crop_report(fine.shape, 4)
    assert note is not None and "2 row(s)" in note and "3 column(s)" in note
    assert crop_report((8, 8), 4) is None
    with pytest.raises(CoarsenError, match="no whole blocks"):
        block_reduce(np.ones((3, 3), np.float32), 4)


def test_channel_width_aggregates_by_max_so_the_river_survives():
    """Mean would thin a 30 m river down to 7.5 m in a 4x4 block. Max keeps it."""
    fine = np.zeros((4, 4), dtype=np.float32)
    fine[2, :] = 30.0  # the river crosses the block in one fine row
    assert block_reduce(fine, 4, "max")[0, 0] == 30.0
    assert block_reduce(fine, 4, "mean")[0, 0] == pytest.approx(7.5)


def test_scenario_indices_follow_the_resolution():
    scn = Scenario(
        inflows=[Inflow(cell=(5, 9), hydrograph=[(0.0, 1.0)])],
        structures=[
            Structure(
                name="d",
                cells=[(8, 8), (8, 9), (9, 8)],
                crest_m=100.0,
                release_rule="fixed",
                release_m3_s=1.0,
                pool=(4, 4, 7, 7),
                outlet=(12, 12, 13, 13),
            )
        ],
    )

    out = coarsen_scenario(scn, 4)

    assert out.inflows[0].cell == (1, 2)
    s = out.structures[0]
    assert s.cells == [(2, 2)]  # three fine cells collapse into one coarse cell
    assert s.pool == (1, 1, 1, 1)
    assert s.outlet == (3, 3, 3, 3)
    assert scn.inflows[0].cell == (5, 9)  # the original is untouched


def test_indices_outside_the_resolved_domain_are_refused():
    scn = Scenario(inflows=[Inflow(cell=(9, 1), hydrograph=[(0.0, 1.0)])])
    check_indices(scn, (10, 10))
    with pytest.raises(CoarsenError, match=r"\[\[inflow\]\] #0"):
        check_indices(scn, (4, 4))

    dam = Scenario(
        structures=[Structure(name="dam", cells=[(30, 1)], crest_m=10.0)],
    )
    with pytest.raises(CoarsenError, match="structure 'dam'"):
        check_indices(dam, (8, 8))


def test_a_coarsened_run_balances_mass_and_records_its_resolution(tmp_path):
    ny = nx = 32
    yy, xx = np.mgrid[0:ny, 0:nx]
    bed = (50.0 + 0.02 * ((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2)).astype(np.float32)
    scn = Scenario(
        name="coarse_bowl",
        dx=10.0,
        coarsen=4,
        end_time=600.0,
        output_every=300.0,
        dt_max=10.0,
        rain_mm_hr=100.0,
        rain_duration=600.0,
        inflows=[Inflow(cell=(16, 16), hydrograph=[(0.0, 2.0), (600.0, 2.0)])],
    )

    ledger = run_simulation(scn, bed, tmp_path / "c.zarr", device="cpu", verbose=False)

    ds = xr.open_zarr(tmp_path / "c.zarr", consolidated=False)
    assert ds["depth"].shape[1:] == (8, 8)  # 32 / 4
    assert ds.attrs["dx"] == 40.0  # 10 m tiles, run at 4x
    assert ds.attrs["coarsen"] == 4
    assert ledger.max_rel_error < MASS_GATE
    # The coarse bed is the block mean of the fine bed, cell for cell.
    np.testing.assert_allclose(ds["bed"].values, block_reduce(bed, 4, "mean"), rtol=1e-6, atol=1e-4)
