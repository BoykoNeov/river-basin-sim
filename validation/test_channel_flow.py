"""Open-boundary validation (M3, HANDOFF §8/§10) on Warp's CPU backend.

The open-outflow sink banks exactly what it removes from ``h`` into ``loss_cum``,
so the mass residual is ~0 *by construction* -- the gate is a bookkeeping check
here, not a physics one. These tests therefore add **independent** checks the
gate cannot make:

* **Drain + non-negativity** on a moderately steep bed -- water actually leaves
  (``outflow -> stored volume``, ``h -> ~0``) and depth never goes negative. The
  non-negativity is the discriminating proof: a boundary-*face* implementation
  would drain edge cells past zero here (the M1 limiter never touches edge faces).
* **Manning normal depth** in a mild steady channel -- inflow at the head, open
  outflow at the toe; at steady state the interior depth matches the analytical
  wide-channel normal depth within a loose band (LI drops advection, so exactness
  is not expected).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State
from solver.io.config import Inflow
from solver.processes.inflow import InflowInjector

wp.init()
DEV = "cpu"

_CLOSED_BUT_EAST = {"east": "open", "west": "closed", "north": "closed", "south": "closed"}


def test_open_edge_drains_and_stays_nonnegative():
    """Moderately steep basin, east edge open: water drains, depth stays >= 0.

    This is the discriminating test the advisor flagged -- the capped sink cannot
    over-drain, so ``h.min() >= 0`` even where the flux is large. A boundary-face
    approach (unlimited at the edge) would go negative here.
    """
    ny, nx, dx = 20, 40, 10.0
    xx = np.broadcast_to(np.arange(nx), (ny, nx))
    slope = 0.02  # 2% -- moderately steep; high at west, low at east
    bed = ((nx - 1 - xx) * dx * slope).astype(np.float32)
    st = State.from_bed(bed, dx=dx, depth=0.5, manning=0.03, device=DEV)
    st.set_open_boundaries(_CLOSED_BUT_EAST)
    ledger = MassLedger.from_state(st)
    v0 = ledger.v0

    t = 0.0
    for _ in range(2000):
        dt = compute_dt(st, alpha=0.7, dt_max=5.0)
        step(st, dt=dt)
        t += dt
    rec = ledger.record(st, t)

    h = st.h.numpy()
    assert np.isfinite(h).all()
    assert h.min() >= -1e-6, f"open BC over-drained to {h.min():.3e} m"
    assert ledger.max_rel_error < MASS_GATE, f"loss bookkeeping off: {ledger.max_rel_error:.2e}"
    # Water genuinely left the domain (not merely rearranged).
    assert rec.outflow_cum > 0.5 * v0
    assert rec.volume < 0.5 * v0


def test_channel_reaches_manning_normal_depth():
    """Mild 1-D channel, steady inflow -> open outflow: interior depth ~ normal.

    Wide-channel Manning normal depth for unit-width discharge ``q = Q/dx`` on
    slope ``S``: ``h_n = (q n / sqrt(S))^(3/5)``. LI is diffusive, so we only
    require the steady interior median within [0.5, 2.0] * h_n.
    """
    nx, dx = 80, 10.0
    slope, n, discharge = 0.001, 0.03, 2.0  # mild 0.1% slope
    bed = ((nx - 1 - np.arange(nx)) * dx * slope).reshape(1, nx).astype(np.float32)
    st = State.from_bed(bed, dx=dx, depth=0.0, manning=n, device=DEV)
    st.set_open_boundaries(_CLOSED_BUT_EAST)
    inj = InflowInjector(
        [Inflow(cell=(0, 0), hydrograph=[(0.0, discharge), (1.0e9, discharge)])], st.grid, DEV
    )
    ledger = MassLedger.from_state(st)

    t = 0.0
    for _ in range(4000):
        dt = compute_dt(st, alpha=0.7, dt_max=10.0)
        ledger.add_inflow(inj.apply(st, t, dt))
        step(st, dt=dt)
        t += dt
    rec = ledger.record(st, t)

    q_w = discharge / dx  # discharge per unit width (m^2/s)
    h_n = (q_w * n / math.sqrt(slope)) ** 0.6
    h = st.h.numpy()[0]
    interior = h[5 : nx - 5]  # skip the head-injection cell and the edge drawdown
    median = float(np.median(interior))

    print(
        f"\n[channel] h_normal={h_n:.3f}m  interior_median={median:.3f}m  mass={rec.rel_error:.2e}"
    )
    assert np.isfinite(h).all()
    assert ledger.max_rel_error < MASS_GATE
    assert rec.outflow_cum > 0.0  # steady flow is leaving the toe
    assert 0.5 * h_n < median < 2.0 * h_n, f"interior depth {median:.3f} far from normal {h_n:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
