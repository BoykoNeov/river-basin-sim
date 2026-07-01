"""Unit tests for the local-inertial kernels (M1).

Run on Warp's **CPU** backend so CI needs no GPU (CLAUDE.md). These are small,
hand-checkable cases: rest state, flux direction, and step-level mass balance.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from solver.core.local_inertial import compute_dt, step
from solver.core.state import State

wp.init()

DEV = "cpu"
N = 0.03  # Manning n


def test_rest_state_stays_at_rest():
    """Uniform depth on a flat bed: no surface gradient, so nothing moves."""
    bed = np.zeros((8, 8), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=1.5, manning=N, device=DEV)
    m0 = float(st.h.numpy().sum())
    for _ in range(20):
        step(st, dt=1.0)
    h = st.h.numpy()
    assert np.allclose(h, 1.5, atol=1e-6)
    assert abs(float(h.sum()) - m0) < 1e-4
    assert np.abs(st.qx.numpy()).max() < 1e-6
    assert np.abs(st.qy.numpy()).max() < 1e-6


def test_flux_flows_downhill_in_x():
    """Two cells, higher depth on the left -> discharge on the shared face is +x."""
    bed = np.zeros((1, 2), dtype=np.float32)
    depth = np.array([[2.0, 1.0]], dtype=np.float32)
    st = State.from_bed(bed, dx=1.0, depth=depth, manning=N, device=DEV)
    step(st, dt=0.01)
    qx = st.qx.numpy()
    assert qx[0, 0] == 0.0 and qx[0, 2] == 0.0  # closed edges
    assert qx[0, 1] > 0.0  # interior face flows left -> right (+x)


def test_single_step_conserves_mass_closed():
    """Closed domain, no rain: continuity is a pure flux divergence -> mass exact."""
    rng = np.random.default_rng(0)
    bed = rng.uniform(0.0, 5.0, size=(6, 7)).astype(np.float32)
    depth = rng.uniform(0.5, 3.0, size=(6, 7)).astype(np.float32)
    st = State.from_bed(bed, dx=5.0, depth=depth, manning=N, device=DEV)
    m0 = float(st.h.numpy().astype(np.float64).sum())
    for _ in range(10):
        dt = compute_dt(st, alpha=0.5, dt_max=5.0)
        step(st, dt=dt)
    m1 = float(st.h.numpy().astype(np.float64).sum())
    # No sources/sinks and closed BCs -> depth-sum invariant to float round-off.
    assert abs(m1 - m0) / m0 < 1e-6
    assert np.isfinite(st.h.numpy()).all()


def test_rainfall_adds_expected_volume():
    """Uniform rain on a closed basin adds rate*area*time of water, no leak."""
    bed = np.zeros((5, 5), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=0.1, manning=N, device=DEV)
    rate_mm_hr = 36.0
    rain = rate_mm_hr / 1000.0 / 3600.0  # m/s = 1e-5
    dt, nsteps = 10.0, 30
    h0 = float(st.h.numpy().astype(np.float64).sum())
    for _ in range(nsteps):
        step(st, dt=dt, rain=rain)
    h1 = float(st.h.numpy().astype(np.float64).sum())
    expected = rain * dt * nsteps * st.grid.n_cells  # depth-sum increase
    # Tolerance is float32 field quantization of a small per-step source, not the
    # solver's mass gate: the <1e-6 gate (massbalance) normalizes by total volume,
    # where this same 1.3e-6 absolute error is ~5e-7. Here we normalize by the
    # (much smaller) increment, so the honest float32 band is ~1e-4.
    assert abs((h1 - h0) - expected) / expected < 1e-4


def test_flux_limiter_conserves_mass_and_forbids_negative_depth():
    """Thin water on a steep bed drives the scheme out of regime.

    Without the limiter this over-drains cells to large negative depths and blows
    up. With it, depths stay non-negative and mass is conserved exactly (the
    limiter only rescales shared face fluxes) -- the closed no-source residual
    must stay under the gate.
    """
    from solver.core.massbalance import MASS_GATE, MassLedger

    ny = nx = 24
    yy = np.arange(ny)[:, None] * np.ones((1, nx))
    bed = (yy * 5.0).astype(np.float32)  # 5 m drop per row -> steep
    st = State.from_bed(bed, dx=10.0, depth=0.005, manning=0.03, device=DEV)  # 5 mm sheet
    ledger = MassLedger.from_state(st)

    t = 0.0
    for _ in range(80):
        dt = compute_dt(st, alpha=0.7, dt_max=5.0)
        step(st, dt=dt)  # limiter on by default
        t += dt
    ledger.record(st, t)

    h = st.h.numpy()
    assert np.isfinite(h).all()
    assert h.min() > -1e-6, f"negative depth {h.min():.3e} despite limiter"
    assert ledger.max_rel_error < MASS_GATE, f"limiter leaked mass {ledger.max_rel_error:.2e}"


def test_manning_field_is_honored():
    """A rougher n field damps the flux harder (the semi-implicit denominator).

    On a flat bed with uniform depth (zero surface gradient, so no pressure
    driving) a pre-seeded discharge is decayed purely by friction:
    ``q_new = q / (1 + g*dt*n^2*|q|/h^(7/3))``. A larger ``n`` -> larger denominator
    -> smaller surviving discharge. This isolates friction from the inertial
    sloshing a closed basin would otherwise show.
    """

    def one_friction_step(n: float) -> float:
        bed = np.zeros((1, 2), dtype=np.float32)
        st = State.from_bed(bed, dx=1.0, depth=1.0, manning=n, device=DEV)
        st.qx = wp.array(np.array([[0.0, 5.0, 0.0]], dtype=np.float32), device=DEV)
        step(st, dt=0.01, limit=False)  # limiter off: isolate the friction term
        return float(st.qx.numpy()[0, 1])

    q_smooth = one_friction_step(0.02)
    q_rough = one_friction_step(0.20)
    assert 0.0 < q_rough < q_smooth < 5.0


def test_compute_dt_is_deterministic_and_bounded():
    bed = np.zeros((4, 4), dtype=np.float32)
    st = State.from_bed(bed, dx=10.0, depth=4.0, device=DEV)
    dt1 = compute_dt(st, alpha=0.7, dt_max=30.0)
    dt2 = compute_dt(st, alpha=0.7, dt_max=30.0)
    assert dt1 == dt2  # same state -> identical dt (determinism)
    # dt = 0.7 * 10 / sqrt(9.81 * 4) ~= 1.117 s, under dt_max
    assert 1.0 < dt1 < 1.2
