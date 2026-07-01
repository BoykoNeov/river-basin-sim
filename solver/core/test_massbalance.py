"""Mass-balance ledger tests (M1) -- the <1e-6 credibility gate, on CPU."""

from __future__ import annotations

import numpy as np
import pytest
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


def test_infiltration_is_capped_and_banked():
    """Infiltration removes at most the available depth and banks it in loss_cum.

    A single big-rate step over a shallow sheet must drain each cell to exactly 0
    (never negative), and the removed volume must equal the ledger outflow.
    """
    bed = np.zeros((4, 4), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=0.01, device=DEV)  # 1 cm sheet
    st.set_infiltration(np.full((4, 4), 1.0, dtype=np.float32))  # 1 m/s -> huge
    ledger = MassLedger.from_state(st)
    step(st, dt=1.0)  # infil*dt = 1 m >> 0.01 m available -> fully drained
    ledger.record(st, 1.0)

    h = st.h.numpy()
    assert h.min() >= 0.0 and float(h.max()) < 1e-7  # emptied, non-negative
    removed = st.loss_volume(st.grid.cell_area)
    assert removed == pytest.approx(0.01 * st.grid.cell_area * st.grid.n_cells, rel=1e-5)
    assert ledger.series[-1].outflow_cum == pytest.approx(removed, rel=1e-6)


def test_infiltration_uncapped_rate_is_exact():
    """Uncapped infiltration removes exactly rate*time*area (independent of the gate).

    The mass gate is ~0 for any sink by construction (loss_cum mirrors h), so it
    can't check the *rate*. Here the cells never run dry -> the removed volume is
    the pure infiltration rate, catching a mis-scaled kernel.

    Note: use *shallow* water. Banking is exact (loss_cum == the depth h lost), but
    h loses ``rate*dt`` only to float32 precision -- and that is worst when
    ``h >> rate*dt`` (large h -> coarse ULP swallows the tiny decrement). Shallow h
    keeps the ULP fine so the rate reads true."""
    rate = 5.0e-5  # m/s
    st = State.from_bed(np.zeros((5, 5), dtype=np.float32), dx=10.0, depth=0.3, device=DEV)
    st.set_infiltration(np.full((5, 5), rate, dtype=np.float32))
    dt, nsteps = 2.0, 20  # removes 2e-3 m << 0.3 m -> uncapped
    for _ in range(nsteps):
        step(st, dt=dt)
    removed = st.loss_volume(st.grid.cell_area)
    expected = rate * dt * nsteps * st.grid.cell_area * st.grid.n_cells
    assert removed == pytest.approx(expected, rel=1e-3)


def test_rain_and_infiltration_balance_to_gate():
    """Uniform rain into a closed basin with a partial infiltration sink: the
    residual (inflow - infiltration_outflow - dV) stays under the <1e-6 gate.

    The infiltration sink is float64-exact by construction, so the residual here
    is really the float32 rain-accumulation floor -- kept well under the gate by a
    realistic (non-thin) stored volume, the same regime as the M1 rain test."""
    bed = np.zeros((16, 16), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=0.5, device=DEV)
    st.set_infiltration(np.full((16, 16), 5.0 / 1000.0 / 3600.0, dtype=np.float32))  # 5 mm/hr
    ledger = MassLedger.from_state(st)
    rain = 50.0 / 1000.0 / 3600.0  # 50 mm/hr

    t = 0.0
    for _ in range(60):
        dt = compute_dt(st, alpha=0.5, dt_max=5.0)
        step(st, dt=dt, rain=rain)
        ledger.add_rain_step(rain, dt, st.grid.n_cells)
        t += dt
    ledger.record(st, t)

    assert ledger.max_rel_error < MASS_GATE
    assert ledger.series[-1].outflow_cum > 0.0  # the sink actually removed water


def test_rain_field_adds_expected_volume():
    """A spatial rain field adds sum(rate)*area*time and balances to the gate."""
    bed = np.zeros((8, 8), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=0.1, device=DEV)
    # A ramp field so the pattern is non-uniform (per-cell mm/hr -> m/s).
    rate_mm_hr = (np.arange(64, dtype=np.float32).reshape(8, 8) + 1.0) * 2.0
    rain_m_s = (rate_mm_hr / 1000.0 / 3600.0).astype(np.float32)
    st.set_rain_field(rain_m_s)
    ledger = MassLedger.from_state(st)
    rain_sum_m_s = float(rain_m_s.astype(np.float64).sum())

    dt, nsteps = 5.0, 40
    for _ in range(nsteps):
        step(st, dt=dt, rain_scale=1.0)
        ledger.add_inflow(rain_sum_m_s * dt * st.grid.cell_area)
    ledger.record(st, dt * nsteps)
    assert ledger.max_rel_error < MASS_GATE


def test_kahan_beats_naive_sum():
    """The compensated accumulator recovers precision a naive float sum loses."""
    from solver.core.massbalance import _Kahan

    k = _Kahan()
    naive = 0.0
    for _ in range(1_000_000):
        k.add(1e-8)
        naive += 1e-8
    assert abs(k.total - 0.01) < abs(naive - 0.01) or abs(k.total - 0.01) < 1e-12
