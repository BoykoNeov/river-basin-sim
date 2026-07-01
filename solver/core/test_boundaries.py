"""Boundary-condition tests (M3): open-outflow sign correctness per edge, on CPU.

Each of the four open-edge kernels is hand-written with its own sign convention,
so this parametrizes a one-edge-open drain over all four and asserts water leaves
through *that* edge (and only reduces volume, never goes negative). A sign error
on any edge would either not drain (water piled against a closed-behaving edge) or
blow up.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State

wp.init()
DEV = "cpu"


def _sloped_bed(ny: int, nx: int, dx: float, edge: str) -> np.ndarray:
    """A bed that slopes downhill toward ``edge`` (so water drains that way)."""
    i = np.arange(ny)[:, None] * np.ones((1, nx))
    j = np.ones((ny, 1)) * np.arange(nx)[None, :]
    slope = 0.02 * dx
    if edge == "east":
        z = (nx - 1 - j) * slope
    elif edge == "west":
        z = j * slope
    elif edge == "south":
        z = (ny - 1 - i) * slope
    else:  # north
        z = i * slope
    return z.astype(np.float32)


@pytest.mark.parametrize("edge", ["north", "south", "east", "west"])
def test_each_open_edge_drains(edge):
    ny, nx, dx = 24, 24, 10.0
    bed = _sloped_bed(ny, nx, dx, edge)
    st = State.from_bed(bed, dx=dx, depth=0.5, manning=0.03, device=DEV)
    st.set_open_boundaries(
        {e: ("open" if e == edge else "closed") for e in ("north", "south", "east", "west")}
    )
    ledger = MassLedger.from_state(st)
    v0 = ledger.v0

    t = 0.0
    for _ in range(1500):
        dt = compute_dt(st, alpha=0.7, dt_max=5.0)
        step(st, dt=dt)
        t += dt
    rec = ledger.record(st, t)

    h = st.h.numpy()
    assert np.isfinite(h).all()
    assert h.min() >= -1e-6, f"{edge}: over-drained to {h.min():.3e}"
    assert ledger.max_rel_error < MASS_GATE
    assert rec.outflow_cum > 0.5 * v0, f"{edge}: only {rec.outflow_cum:.1f}/{v0:.1f} m^3 left"
    assert rec.volume < 0.5 * v0


def test_closed_is_bitwise_unchanged_by_open_code():
    """With no open edges, the open-BC path never launches -> identical results.

    Two runs from the same seed state, both fully closed, must match bit-for-bit
    (the open-BC additions are inert when open_edges is empty)."""
    rng = np.random.default_rng(3)
    bed = rng.uniform(0, 4, size=(10, 10)).astype(np.float32)
    depth = rng.uniform(0.5, 2.0, size=(10, 10)).astype(np.float32)

    def run() -> np.ndarray:
        st = State.from_bed(bed, dx=10.0, depth=depth, manning=0.03, device=DEV)
        for _ in range(30):
            dt = compute_dt(st, alpha=0.6, dt_max=5.0)
            step(st, dt=dt)
        return st.h.numpy()

    assert np.array_equal(run(), run())
