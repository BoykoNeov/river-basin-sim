"""Mass-balance ledger tests (M1) -- the <1e-6 credibility gate, on CPU."""

from __future__ import annotations

import numpy as np
import warp as wp

from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State

wp.init()

DEV = "cpu"


def test_closed_no_source_conserves_to_gate():
    """Sloshing water in a closed box with no rain: residual stays under the gate."""
    rng = np.random.default_rng(1)
    bed = rng.uniform(0.0, 3.0, size=(12, 10)).astype(np.float32)
    depth = rng.uniform(1.0, 4.0, size=(12, 10)).astype(np.float32)
    st = State.from_bed(bed, dx=8.0, depth=depth, manning=0.03, device=DEV)
    ledger = MassLedger.from_state(st)

    t = 0.0
    for _ in range(50):
        dt = compute_dt(st, alpha=0.5, dt_max=5.0)
        step(st, dt=dt)
        t += dt
    ledger.record(st, t)

    assert ledger.max_rel_error < MASS_GATE


def test_uniform_rain_balances_to_gate():
    """Rain on a closed basin: inflow_cum matches stored-volume rise to the gate."""
    bed = np.zeros((16, 16), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=0.2, manning=0.03, device=DEV)
    ledger = MassLedger.from_state(st)
    rain = 50.0 / 1000.0 / 3600.0  # 50 mm/hr -> m/s

    t = 0.0
    for _ in range(40):
        dt = compute_dt(st, alpha=0.5, dt_max=5.0)
        step(st, dt=dt, rain=rain)
        ledger.add_rain_step(rain, dt, st.grid.n_cells)
        t += dt
    ledger.record(st, t)

    assert ledger.max_rel_error < MASS_GATE
    # Sanity: volume actually grew by ~inflow.
    assert ledger.series[-1].volume > ledger.v0


def test_kahan_beats_naive_sum():
    """The compensated accumulator recovers precision a naive float sum loses."""
    from solver.core.massbalance import _Kahan

    k = _Kahan()
    naive = 0.0
    for _ in range(1_000_000):
        k.add(1e-8)
        naive += 1e-8
    assert abs(k.total - 0.01) < abs(naive - 0.01) or abs(k.total - 0.01) < 1e-12
