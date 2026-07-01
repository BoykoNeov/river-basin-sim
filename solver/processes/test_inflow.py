"""Inflow-hydrograph tests (M3): injected volume + mass balance, on CPU."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core.grid import Grid
from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State
from solver.io.config import Inflow
from solver.processes.inflow import InflowInjector

wp.init()
DEV = "cpu"


def test_breakpoints_are_sorted_union():
    inj = InflowInjector(
        [
            Inflow(cell=(1, 1), hydrograph=[(0.0, 1.0), (600.0, 1.0)]),
            Inflow(cell=(2, 2), hydrograph=[(300.0, 0.0), (900.0, 2.0)]),
        ],
        Grid(ny=4, nx=4, dx=10.0),
        DEV,
    )
    assert inj.breakpoints() == [0.0, 300.0, 600.0, 900.0]


def test_out_of_bounds_cell_rejected():
    with pytest.raises(ValueError, match="outside"):
        InflowInjector(
            [Inflow(cell=(9, 0), hydrograph=[(0.0, 1.0)])], Grid(ny=4, nx=4, dx=10.0), DEV
        )


def test_injected_volume_matches_hydrograph():
    """A constant 3 m^3/s inflow for 100 s adds ~300 m^3 to the target cell."""
    grid = Grid(ny=5, nx=5, dx=10.0)
    st = State.from_bed(np.zeros((5, 5), dtype=np.float32), dx=10.0, device=DEV)
    inj = InflowInjector([Inflow(cell=(2, 2), hydrograph=[(0.0, 3.0), (100.0, 3.0)])], grid, DEV)

    total = 0.0
    t, dt = 0.0, 10.0
    for _ in range(10):
        total += inj.apply(st, t, dt)
        t += dt
    assert total == pytest.approx(300.0, rel=1e-4)
    # All the water landed in cell (2,2): volume = depth * area.
    assert float(st.h.numpy()[2, 2]) * grid.cell_area == pytest.approx(300.0, rel=1e-4)


def test_closed_basin_with_inflow_balances_to_gate():
    """Inflow into a closed flat basin: inflow_cum tracks the volume rise (<1e-6)."""
    grid = Grid(ny=12, nx=12, dx=10.0)
    st = State.from_bed(np.zeros((12, 12), dtype=np.float32), dx=10.0, depth=0.05, device=DEV)
    inj = InflowInjector(
        [Inflow(cell=(6, 6), hydrograph=[(0.0, 0.0), (200.0, 4.0), (600.0, 0.0)])], grid, DEV
    )
    ledger = MassLedger.from_state(st)

    t = 0.0
    for _ in range(120):
        dt = compute_dt(st, alpha=0.5, dt_max=5.0)
        ledger.add_inflow(inj.apply(st, t, dt))
        step(st, dt=dt)
        t += dt
    ledger.record(st, t)

    assert ledger.max_rel_error < MASS_GATE
    assert ledger.series[-1].inflow_cum > 0.0  # water actually entered
