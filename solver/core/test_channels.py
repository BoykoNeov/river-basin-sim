"""Sub-grid channel unit tests (M6 plan §3) -- geometry, storage curve, the seam.

The physics gates live in ``validation/test_subgrid_channel.py``; these are the
algebraic and structural ones: the storage curve is continuous, monotone and
invertible, the kernel agrees with the host reference, bad geometry is refused,
the timestep sees the water *column*, and -- the seam gate -- arming channels with
zero width reproduces the M1 scheme **bitwise**.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from solver.core.channels import (
    ChannelError,
    arm_channels,
    eta_from_h,
    h_from_eta,
    validate_geometry,
)
from solver.core.grid import GRAVITY, Grid
from solver.core.local_inertial import compute_dt, compute_eta_channels, step
from solver.core.state import State

wp.init()
DEV = "cpu"


def _flat_state(ny=6, nx=10, dx=50.0, depth=0.0, bed=None) -> State:
    if bed is None:
        bed = np.full((ny, nx), 10.0, dtype=np.float32)
    return State.from_bed(bed, dx=dx, depth=depth, manning=0.035, device=DEV)


# --- the storage curve -------------------------------------------------------


def test_storage_curve_is_continuous_monotone_and_invertible():
    dx, w, d, z = 100.0, 20.0, 3.0, 12.0
    h_bf = w * d / dx  # 0.6 m cell-mean at bank full

    # Continuity at bank full: both branches meet at the floodplain bed.
    assert eta_from_h(h_bf, z, w, d, dx) == pytest.approx(z, abs=1e-12)
    assert eta_from_h(h_bf - 1e-9, z, w, d, dx) == pytest.approx(z, abs=1e-6)
    assert eta_from_h(h_bf + 1e-9, z, w, d, dx) == pytest.approx(z, abs=1e-6)

    # Below bank full the channel amplifies depth by dx/w; above it, it does not.
    assert eta_from_h(h_bf / 2, z, w, d, dx) == pytest.approx(z - d / 2)
    assert eta_from_h(h_bf + 0.4, z, w, d, dx) == pytest.approx(z + 0.4)

    h = np.linspace(0.0, 5.0, 501)
    eta = eta_from_h(h, z, w, d, dx)
    assert np.all(np.diff(eta) > 0.0)  # strictly monotone
    np.testing.assert_allclose(h_from_eta(eta, z, w, d, dx), h, atol=1e-12)

    # An empty channel puts the surface at the channel bed, not the floodplain bed.
    assert eta_from_h(0.0, z, w, d, dx) == pytest.approx(z - d)


def test_no_channel_is_exactly_the_pre_m6_relation():
    h = np.linspace(0.0, 3.0, 64).astype(np.float64)
    z = np.full_like(h, 7.25)
    zero = np.zeros_like(h)
    np.testing.assert_array_equal(eta_from_h(h, z, zero, zero, 30.0), z + h)


def test_kernel_storage_curve_matches_the_host_reference():
    """The wp.func and the numpy reference must not drift apart."""
    ny, nx, dx = 4, 16, 100.0
    rng = np.random.default_rng(7)
    h = rng.uniform(0.0, 2.0, (ny, nx)).astype(np.float32)
    z = rng.uniform(5.0, 15.0, (ny, nx)).astype(np.float32)
    w = np.where(rng.random((ny, nx)) < 0.5, 0.0, rng.uniform(5.0, 40.0, (ny, nx))).astype(
        np.float32
    )
    d = np.where(w > 0, rng.uniform(0.5, 4.0, (ny, nx)), 0.0).astype(np.float32)

    st = _flat_state(ny, nx, dx, bed=z)
    st.h = wp.array(h, dtype=wp.float32, device=DEV)
    chan = arm_channels(st, w, d)
    wp.launch(
        compute_eta_channels,
        dim=(ny, nx),
        inputs=[st.h, st.z, chan.w, chan.d, dx, st.eta],
        device=DEV,
    )
    np.testing.assert_allclose(st.eta.numpy(), eta_from_h(h, z, w, d, dx), rtol=1e-5, atol=1e-5)


# --- geometry validation -----------------------------------------------------


def test_geometry_that_cannot_mean_anything_is_refused():
    grid = Grid(ny=3, nx=4, dx=50.0)
    ok_w = np.full((3, 4), 10.0, dtype=np.float32)
    ok_d = np.full((3, 4), 2.0, dtype=np.float32)
    ok_n = np.full((3, 4), 0.03, dtype=np.float32)

    wide = ok_w.copy()
    wide[1, 1] = 60.0  # wider than the 50 m cell -- not sub-grid at all
    with pytest.raises(ChannelError, match="wider than the cell"):
        validate_geometry(wide, ok_d, ok_n, grid)

    half = ok_d.copy()
    half[0, 0] = 0.0  # a width with no bank-full depth
    with pytest.raises(ChannelError, match="width without a bank-full depth"):
        validate_geometry(ok_w, half, ok_n, grid)

    neg = ok_w.copy()
    neg[2, 3] = -1.0
    with pytest.raises(ChannelError, match=">= 0"):
        validate_geometry(neg, ok_d, ok_n, grid)

    with pytest.raises(ChannelError, match="shape"):
        validate_geometry(np.zeros((2, 2), np.float32), ok_d, ok_n, grid)

    bad_n = ok_n.copy()
    bad_n[0, 0] = 0.0
    with pytest.raises(ChannelError, match="manning > 0"):
        validate_geometry(ok_w, ok_d, bad_n, grid)


def test_a_hairline_channel_warns_because_it_will_crawl():
    grid = Grid(ny=2, nx=2, dx=200.0)
    w = np.full((2, 2), 0.01, dtype=np.float32)  # 1:20000 -- a data artifact
    d = np.full((2, 2), 1.0, dtype=np.float32)
    n = np.full((2, 2), 0.03, dtype=np.float32)
    with pytest.warns(UserWarning, match="timestep"):
        validate_geometry(w, d, n, grid)


def test_no_channel_normalises_to_zero_in_both_fields():
    grid = Grid(ny=2, nx=2, dx=50.0)
    w = np.array([[0.0, 10.0], [0.0, 0.0]], dtype=np.float32)
    d = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    n = np.full((2, 2), 0.03, dtype=np.float32)
    w_out, d_out, _ = validate_geometry(w, d, n, grid)
    assert w_out[0, 0] == 0.0 and d_out[0, 0] == 0.0
    assert w_out[0, 1] == 10.0 and d_out[0, 1] == 2.0


def test_arming_defaults_channel_roughness_to_the_floodplain_value():
    st = _flat_state()
    chan = arm_channels(st, np.full(st.grid.shape, 8.0), np.full(st.grid.shape, 1.5))
    np.testing.assert_array_equal(chan.n.numpy(), st.n.numpy())
    assert chan.channel_cells == st.grid.n_cells
    assert "channel cells" in chan.summary()


# --- the seam ----------------------------------------------------------------


def _slope_bed(ny: int, nx: int, dx: float, slope: float) -> np.ndarray:
    xx = np.broadcast_to(np.arange(nx), (ny, nx))
    return ((nx - 1 - xx) * dx * slope + 5.0).astype(np.float32)


def test_zero_width_channels_reproduce_the_m1_scheme_bitwise():
    """The sub-grid path must not perturb the permanent coverage scheme.

    Same terrain and forcing, once through the M1 kernels and once through the
    sub-grid kernels with ``w = 0`` everywhere -- **bit for bit**, not close.
    """
    ny, nx, dx = 8, 24, 40.0
    bed = _slope_bed(ny, nx, dx, 0.004)

    plain = State.from_bed(bed, dx=dx, depth=0.05, manning=0.03, device=DEV)
    armed = State.from_bed(bed, dx=dx, depth=0.05, manning=0.03, device=DEV)
    arm_channels(armed, np.zeros((ny, nx), np.float32), np.zeros((ny, nx), np.float32))

    for _ in range(60):
        dt_p = compute_dt(plain, alpha=0.7, dt_max=5.0)
        dt_a = compute_dt(armed, alpha=0.7, dt_max=5.0)
        assert dt_p == dt_a
        step(plain, dt=dt_p, rain=1e-5)
        step(armed, dt=dt_a, rain=1e-5)

    np.testing.assert_array_equal(armed.h.numpy(), plain.h.numpy())
    np.testing.assert_array_equal(armed.qx.numpy(), plain.qx.numpy())
    np.testing.assert_array_equal(armed.qy.numpy(), plain.qy.numpy())


def test_timestep_is_set_by_the_water_column_not_the_cell_mean():
    """A channel concentrates depth by dx/w, and the CFL bound must see it."""
    ny, nx, dx = 4, 8, 200.0
    w, d, h0 = 20.0, 3.0, 0.05  # in-bank: column = h*dx/w = 0.5 m
    plain = _flat_state(ny, nx, dx, depth=h0)
    armed = _flat_state(ny, nx, dx, depth=h0)
    arm_channels(armed, np.full((ny, nx), w, np.float32), np.full((ny, nx), d, np.float32))

    dt_plain = compute_dt(plain, alpha=0.7, dt_max=1e6)
    dt_armed = compute_dt(armed, alpha=0.7, dt_max=1e6)

    assert dt_plain == pytest.approx(0.7 * dx / math.sqrt(GRAVITY * h0), rel=1e-5)
    assert dt_armed == pytest.approx(0.7 * dx / math.sqrt(GRAVITY * h0 * dx / w), rel=1e-5)
    assert dt_armed < dt_plain / 3.0  # sqrt(dx/w) = sqrt(10)


def test_a_closed_channelled_box_conserves_mass_exactly():
    """Continuity is untouched by the storage curve: volume is a pure divergence."""
    ny, nx, dx = 10, 20, 50.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    bed = (20.0 + 0.02 * ((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2)).astype(np.float32)
    st = State.from_bed(bed, dx=dx, depth=0.2, manning=0.03, device=DEV)
    # A channel down the middle row, dead-ending nowhere -- water sloshes into it.
    w = np.zeros((ny, nx), np.float32)
    w[ny // 2, :] = 15.0
    d = np.zeros((ny, nx), np.float32)
    d[ny // 2, :] = 2.0
    arm_channels(st, w, d)

    v0 = float(st.h.numpy().astype(np.float64).sum())
    for _ in range(300):
        step(st, dt=compute_dt(st, alpha=0.7, dt_max=5.0))
    h = st.h.numpy()
    v1 = float(h.astype(np.float64).sum())

    assert np.isfinite(h).all()
    assert h.min() >= -1e-9
    assert abs(v1 - v0) / v0 < 1e-6
