"""Inflow-hydrograph tests (M3): injected volume + mass balance, on CPU.

The compensation gates at the bottom are **ratios, not thresholds** (the discipline
`solver/core/test_sources.py` set): a threshold passes just as happily if the
compensation array is never written, so it cannot tell a working fix from one that
was compiled away. Each runs the same configuration with `compensated=False` and
`True` and gates the improvement.

They deliberately gate the *accumulation*, not a stepped run's mass residual. That
is not a shortcut, it is what the measurements say: when nothing else writes `h`,
the discarded low bits stay correlated step to step and the uncompensated drift is
**systematic** (489x at 1500 steps). Once continuity rewrites `h` every step the
low bits decorrelate and the same drift becomes a random walk, so an A/B on the
ledger residual of a flowing fixture comes out as noise -- measured at 0.8x, 2.8x
and 7.0x over 300/600/1200 steps, i.e. sometimes "worse". Gating that would be
gating a coin flip. The real-scenario numbers are in
`docs/plans/point-source-compensation.md`; what a unit test can hold is the
arithmetic.
"""

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


def test_duplicate_cell_rejected():
    """Two entries on the same cell would race (non-atomic h += ...) -> rejected."""
    with pytest.raises(ValueError, match="duplicate"):
        InflowInjector(
            [
                Inflow(cell=(2, 2), hydrograph=[(0.0, 1.0), (100.0, 1.0)]),
                Inflow(cell=(2, 2), hydrograph=[(0.0, 2.0), (100.0, 2.0)]),
            ],
            Grid(ny=4, nx=4, dx=10.0),
            DEV,
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


# --- Kahan compensation of the point-source add --------------------------------


def _accumulate(compensated: bool, *, q: float, dt: float, nsteps: int, depth: float = 1.0):
    """Inject ``q`` for ``nsteps`` steps with nothing else touching ``h``.

    Returns ``(volume_error, requested_volume)``: how far the field's own rise ends
    up from the volume the injector reported to the ledger. No :func:`step` call, so
    the only arithmetic in the answer is the source add itself.
    """
    grid = Grid(ny=3, nx=3, dx=100.0)
    st = State.from_bed(np.zeros((3, 3), dtype=np.float32), dx=100.0, depth=depth, device=DEV)
    inj = InflowInjector(
        [Inflow(cell=(1, 1), hydrograph=[(0.0, q), (1e9, q)])], grid, DEV, compensated=compensated
    )
    requested = 0.0
    for k in range(nsteps):
        requested += inj.apply(st, k * dt, dt)
    got = (float(st.h.numpy()[1, 1]) - depth) * grid.cell_area
    return abs(got - requested), requested


def test_point_source_add_is_compensated():
    """A realistic flood increment (0.027 m onto 1 m), 1500 steps: ratio > 100.

    This is `reach_alluvial`'s regime shrunk to a unit test -- 90 m^3/s into a
    100 m cell, which is `Q*dt/A ~ 0.027` m of float32 added onto metre-deep water,
    once per cell per step. Uncompensated the field ends up ~5 m^3 away from the
    405,000 m^3 the ledger was told about; compensated, ~0.01 m^3.
    """
    uncomp, total = _accumulate(False, q=90.0, dt=3.0, nsteps=1500)
    comp, _ = _accumulate(True, q=90.0, dt=3.0, nsteps=1500)
    assert total == pytest.approx(405_000.0, rel=1e-6)
    assert uncomp / comp > 100.0, f"uncompensated {uncomp} vs compensated {comp}"


def test_sub_ulp_point_source_lands_nothing_uncompensated():
    """An increment below half an ulp of ``h``: the sharpest statement of the defect.

    4e-8 m onto 1.0 m is under `eps(1.0)/2 = 5.96e-8`, so every uncompensated add
    rounds straight back to where it started and 2000 steps of inflow land *exactly*
    nothing -- while the ledger banks the full volume. Compensated, the debt
    accumulates until it is representable and the field reaches the analytic total.
    """
    area = Grid(ny=3, nx=3, dx=100.0).cell_area  # `_accumulate`'s own grid
    dt, add = 10.0, 4.0e-8
    q = add * area / dt
    uncomp, total = _accumulate(False, q=q, dt=dt, nsteps=2000)
    comp, _ = _accumulate(True, q=q, dt=dt, nsteps=2000)
    # Uncompensated the error *is* the whole requested volume: nothing landed.
    assert uncomp == pytest.approx(total, rel=1e-9)
    assert uncomp / comp > 100.0


def test_compensation_term_is_nonzero_fast_math_canary():
    """`(t - h) - y` must not be reassociated to zero (cf. the rain canary).

    If a compiler flag or a Warp default ever folds the compensation away this array
    stays all-zero, and every other assertion in this section would be measuring an
    uncompensated add against itself.
    """
    grid = Grid(ny=3, nx=3, dx=100.0)
    st = State.from_bed(np.zeros((3, 3), dtype=np.float32), dx=100.0, depth=1.0, device=DEV)
    inj = InflowInjector([Inflow(cell=(1, 1), hydrograph=[(0.0, 90.0), (1e9, 90.0)])], grid, DEV)
    inj.apply(st, 0.0, 3.0)
    comp = inj.compensation_numpy()
    assert comp is not None and np.any(comp != 0.0), "no compensation banked"


def test_uncompensated_arm_reproduces_the_plain_add():
    """`compensated=False` must be the pre-fix arithmetic, or the ratios mean nothing.

    Checked against a hand-rolled float32 running sum rather than against a
    tolerance: the control arm is only a control if it is bit-for-bit the plain
    `h += Q*dt/area` the fix replaced.
    """
    grid = Grid(ny=3, nx=3, dx=100.0)
    st = State.from_bed(np.zeros((3, 3), dtype=np.float32), dx=100.0, depth=1.0, device=DEV)
    inj = InflowInjector(
        [Inflow(cell=(1, 1), hydrograph=[(0.0, 90.0), (1e9, 90.0)])], grid, DEV, compensated=False
    )
    assert inj.compensation_numpy() is None
    expect = np.float32(1.0)
    for k in range(200):
        vol = inj.apply(st, k * 3.0, 3.0)
        expect = np.float32(expect + np.float32(vol / grid.cell_area))
    assert np.float32(st.h.numpy()[1, 1]) == expect


def test_receded_hydrograph_keeps_carrying_its_debt():
    """A hydrograph that falls to zero must not strand the outstanding compensation.

    The kernel keeps being launched at `Q = 0` (there is no zero-discharge
    shortcut), so the debt is still carried and repaid the moment it becomes
    representable. Gated by running a rise-peak-recede hydrograph with a long dry
    tail and checking the compensated field is still far closer to the banked
    volume than the uncompensated one at the *end* of that tail.
    """
    grid = Grid(ny=3, nx=3, dx=100.0)
    hydro = [(0.0, 0.0), (300.0, 90.0), (900.0, 90.0), (1200.0, 0.0), (6000.0, 0.0)]

    def run(compensated: bool) -> tuple[float, np.ndarray | None]:
        st = State.from_bed(np.zeros((3, 3), dtype=np.float32), dx=100.0, depth=1.0, device=DEV)
        inj = InflowInjector(
            [Inflow(cell=(1, 1), hydrograph=hydro)], grid, DEV, compensated=compensated
        )
        requested = 0.0
        for k in range(2000):  # 2000 x 3 s = 6000 s: 900 s of flow, 5100 s of tail
            requested += inj.apply(st, k * 3.0, 3.0)
        got = (float(st.h.numpy()[1, 1]) - 1.0) * grid.cell_area
        return abs(got - requested), inj.compensation_numpy()

    uncomp, _ = run(False)
    comp, debt = run(True)
    assert uncomp / comp > 20.0, f"uncompensated {uncomp} vs compensated {comp}"
    # The debt is bounded (it is low-order bits of `h`), not growing without limit.
    assert debt is not None and np.all(np.abs(debt) < 1.0e-5)
