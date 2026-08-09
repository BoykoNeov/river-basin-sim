"""Bedload transport + Exner arithmetic (solver.core.sediment).

Pure-kernel tests: no run loop, no scheduler, no scenario. The gates that decide
whether M7 is *right* -- sediment mass conservation, bed-wave celerity, interval
independence -- need a scenario and land later; what is checked here is that each
piece computes what it claims, against hand-computed values and against the code
the milestone must not contradict:

* the shear is the **same Manning form** the momentum update already carries
  (:func:`test_shear_is_the_manning_denominators_own_friction_term`), which is the
  reason this module writes ``tau`` out rather than recovering it as ``(D-1)/dt``;
* a channel face's transport is per unit **channel** width and mixes by ``w/dx``
  (:func:`test_a_channel_face_carries_its_conveyance_fraction`) -- the factor-of-15
  error M7 plan §4 names as the milestone's likeliest physics bug;
* the channel kernels **collapse** onto the plain ones at zero width, bit for bit,
  the way :func:`solver.core.channels.eta_subgrid` collapses onto ``h + z``.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core.friction import manning_denominator, manning_denominator_radius
from solver.core.grid import GRAVITY, H_DRY
from solver.core.sediment import (
    MPM_COEFFICIENT,
    SHIELDS_CRITICAL,
    SUBMERGED_SG,
    accumulate_qs_x,
    accumulate_qs_x_channels,
    accumulate_qs_y,
    bed_celerity,
    capacity_from_flow,
    clear_transport_integral,
    exner_update,
    face_capacity,
    kinematic_shear,
    morphological_courant,
    rebuild_bed,
    shields_from_flow,
    shields_number,
)

wp.init()

DEV = "cpu"


# --- kernels that expose the @wp.func layer to the host ----------------------


@wp.kernel
def _eval_shear(
    q: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    r: wp.array(dtype=wp.float32),
    n: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = kinematic_shear(q[i], h[i], r[i], n[i])


@wp.kernel
def _eval_denominator(
    q: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    r: wp.array(dtype=wp.float32),
    n: wp.array(dtype=wp.float32),
    dt: wp.float32,
    out: wp.array(dtype=wp.float32),
    out_wide: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = manning_denominator_radius(q[i], h[i], r[i], n[i], GRAVITY, dt)
    out_wide[i] = manning_denominator(q[i], h[i], n[i], GRAVITY, dt)


@wp.kernel
def _eval_theta_and_capacity(
    q: wp.array(dtype=wp.float32),
    h: wp.array(dtype=wp.float32),
    r: wp.array(dtype=wp.float32),
    n: wp.array(dtype=wp.float32),
    d50: wp.array(dtype=wp.float32),
    theta: wp.array(dtype=wp.float32),
    qs: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    theta[i] = shields_number(q[i], h[i], r[i], n[i], d50[i])
    qs[i] = face_capacity(q[i], h[i], r[i], n[i], d50[i])


def _f32(a) -> wp.array:
    return wp.array(np.ascontiguousarray(a, dtype=np.float32), dtype=wp.float32, device=DEV)


def _sample_faces(seed: int = 7, count: int = 64):
    """Random but physical face states: depths cm..m, discharges up to a few m^2/s."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.02, 3.0, count)
    q = rng.uniform(-3.0, 3.0, count)
    w = rng.uniform(1.0, 40.0, count)
    r = w * h / (w + 2.0 * h)
    n = rng.uniform(0.02, 0.06, count)
    d50 = rng.uniform(2e-4, 3e-2, count)
    return h, q, r, n, d50


# --- the law -----------------------------------------------------------------


def test_shear_is_the_manning_denominators_own_friction_term():
    """One Manning form, checked rather than shared by fragile algebra (M7 §1.2).

    ``tau/rho = g n^2 q^2 / (h^2 R^(1/3))`` is exactly the friction term the
    semi-implicit denominator carries: ``D = 1 + dt * (tau/rho) * h / (R |q|)``.
    This module writes ``tau`` out directly instead of recovering it from ``D``,
    because ``D - 1`` cancels catastrophically in float32 for small ``dt`` -- so the
    two definitions are tied together here instead. If a future change moves the
    hydraulic radius in one of them, this fails.
    """
    h, q, r, n, _ = _sample_faces()
    count, dt = h.size, 0.5
    tau = wp.zeros(count, dtype=wp.float32, device=DEV)
    den = wp.zeros(count, dtype=wp.float32, device=DEV)
    den_wide = wp.zeros(count, dtype=wp.float32, device=DEV)
    wp.launch(_eval_shear, dim=count, inputs=[_f32(q), _f32(h), _f32(r), _f32(n), tau], device=DEV)
    wp.launch(
        _eval_denominator,
        dim=count,
        inputs=[_f32(q), _f32(h), _f32(r), _f32(n), dt, den, den_wide],
        device=DEV,
    )
    got = 1.0 + dt * tau.numpy().astype(np.float64) * h / (r * np.abs(q))
    assert np.allclose(got, den.numpy().astype(np.float64), rtol=1e-5)

    # ... and the wide-channel limit R = h is the floodplain denominator verbatim.
    tau_wide = wp.zeros(count, dtype=wp.float32, device=DEV)
    wp.launch(
        _eval_shear, dim=count, inputs=[_f32(q), _f32(h), _f32(h), _f32(n), tau_wide], device=DEV
    )
    got_wide = 1.0 + dt * tau_wide.numpy().astype(np.float64) / np.abs(q)
    assert np.allclose(got_wide, den_wide.numpy().astype(np.float64), rtol=1e-5)


def test_shields_and_capacity_match_the_host_reference():
    """The device law and the host reference the gates are written against agree."""
    h, q, r, n, d50 = _sample_faces(seed=11)
    count = h.size
    theta = wp.zeros(count, dtype=wp.float32, device=DEV)
    qs = wp.zeros(count, dtype=wp.float32, device=DEV)
    wp.launch(
        _eval_theta_and_capacity,
        dim=count,
        inputs=[_f32(q), _f32(h), _f32(r), _f32(n), _f32(d50), theta, qs],
        device=DEV,
    )
    assert np.allclose(theta.numpy(), shields_from_flow(q, h, n, d50, r), rtol=1e-5)
    expected = capacity_from_flow(q, h, n, d50, r) * np.sign(q)
    assert np.allclose(qs.numpy(), expected, rtol=1e-4, atol=1e-12)


def test_capacity_is_hand_computable():
    """One arithmetic case, worked by hand, so the constants cannot drift silently."""
    q, h, n, d50 = 2.0, 1.0, 0.03, 0.008
    theta = GRAVITY * n**2 * q**2 / h ** (7.0 / 3.0) / (SUBMERGED_SG * GRAVITY * d50)
    assert theta == pytest.approx(0.27273, rel=1e-4)  # well above 0.047: it moves
    qs = (
        MPM_COEFFICIENT
        * (theta - SHIELDS_CRITICAL) ** 1.5
        * np.sqrt(SUBMERGED_SG * GRAVITY * d50**3)
    )
    assert qs == pytest.approx(2.4699e-3, rel=1e-4)  # m^2/s per unit width
    assert capacity_from_flow(q, h, n, d50) == pytest.approx(qs, rel=1e-9)


def test_below_threshold_the_capacity_is_bit_exact_zero():
    """No motion => no bed change, exactly -- the cheap gate that catches a stray term.

    A velocity-independent term, a sign error or a units error all show up as a
    nonzero capacity here, and the assertion is ``== 0.0``, not ``< tol``.
    """
    d50 = 0.008
    # Solve for the depth that puts theta exactly at threshold at this discharge,
    # then step to either side of it.
    q, n = 1.0, 0.03
    h_c = (GRAVITY * n**2 * q**2 / (SUBMERGED_SG * GRAVITY * d50 * SHIELDS_CRITICAL)) ** (3 / 7)
    below = np.array([h_c * 1.05, h_c * 2.0, 5.0])  # deeper => slower => less shear
    above = np.array([h_c * 0.95, h_c * 0.5])
    count = below.size + above.size
    h = np.concatenate([below, above])
    theta = wp.zeros(count, dtype=wp.float32, device=DEV)
    qs = wp.zeros(count, dtype=wp.float32, device=DEV)
    ones = np.ones(count)
    wp.launch(
        _eval_theta_and_capacity,
        dim=count,
        inputs=[_f32(q * ones), _f32(h), _f32(h), _f32(n * ones), _f32(d50 * ones), theta, qs],
        device=DEV,
    )
    got = qs.numpy()
    assert (got[: below.size] == 0.0).all()
    assert (got[below.size :] > 0.0).all()


def test_a_dry_face_transports_nothing():
    """The wet/dry guard is the scheme's, not a new one (M7 §1.6)."""
    count = 3
    h = np.array([0.0, H_DRY * 0.5, H_DRY * 2.0])
    qs = wp.zeros(count, dtype=wp.float32, device=DEV)
    theta = wp.zeros(count, dtype=wp.float32, device=DEV)
    wp.launch(
        _eval_theta_and_capacity,
        dim=count,
        inputs=[
            _f32(np.full(count, 0.5)), _f32(h), _f32(np.maximum(h, 1e-6)),
            _f32(np.full(count, 0.03)), _f32(np.full(count, 0.008)), theta, qs,
        ],
        device=DEV,
    )  # fmt: skip
    got = qs.numpy()
    assert got[0] == 0.0 and got[1] == 0.0
    assert got[2] > 0.0


# --- accumulation on faces ---------------------------------------------------


def _flat_channel_bed(ny=3, nx=5, slope_per_cell=0.0, base=10.0) -> np.ndarray:
    z = np.full((ny, nx), base, dtype=np.float32)
    z -= slope_per_cell * np.arange(nx, dtype=np.float32)[None, :]
    return z


def _plain_accumulator(ny, nx):
    return (
        wp.zeros((ny, nx + 1), dtype=wp.float32, device=DEV),
        wp.zeros((ny, nx + 1), dtype=wp.float32, device=DEV),
    )


def test_the_face_integral_is_transport_times_elapsed_time():
    """A steady flow for N steps integrates to ``q_s * dt * N`` (M7 §1.3).

    This is what makes the bed move by a time *integral* rather than an instant
    sample scaled by the interval, and it is the property the interval-independence
    gate rests on.
    """
    ny, nx, dt, steps = 3, 5, 4.0, 25
    z = _flat_channel_bed(ny, nx)
    depth, q, n, d50 = 1.2, 2.5, 0.03, 0.008
    eta = _f32(z + depth)
    qx = _f32(np.full((ny, nx + 1), q))
    qs_int, qs_comp = _plain_accumulator(ny, nx)
    for _ in range(steps):
        wp.launch(
            accumulate_qs_x,
            dim=(ny, nx - 1),
            inputs=[
                qx, eta, _f32(z), _f32(np.full((ny, nx), n)), _f32(np.full((ny, nx), d50)),
                dt, qs_int, qs_comp,
            ],
            device=DEV,
        )  # fmt: skip
    expected = capacity_from_flow(q, depth, n, d50) * dt * steps
    got = qs_int.numpy()
    assert np.allclose(got[:, 1:nx], expected, rtol=1e-5)
    assert (got[:, 0] == 0.0).all() and (got[:, nx] == 0.0).all()  # boundary faces untouched


def test_the_transport_integral_is_compensated_not_merely_added():
    """A sub-ulp increment survives, which a plain ``+=`` cannot (the canary, §1.3).

    ``qs_int`` accumulates through :func:`solver.core.sources.kahan_add`, and there
    is no uncompensated twin kernel to take a *ratio* against -- so this canary
    carries the whole weight of that claim, which raises its importance rather than
    lowering it (compare ``test_sources.py``'s docstring: a threshold test passes
    just as happily if the compensation array is never written, or if a future
    fast-math default reassociates it away).

    Seeded at 4 m^2 with ``dt = 1 us`` each increment is ~2.5e-9 against
    ``eps(4) = 4.8e-7``, so uncompensated every add rounds straight back to 4.0 and
    the integral never moves at all.
    """
    ny, nx, dt, steps = 1, 3, 1.0e-6, 8000
    z = _flat_channel_bed(ny, nx)
    depth, q, n, d50 = 1.0, 2.0, 0.03, 0.008
    seed = 4.0
    qs_int, qs_comp = _plain_accumulator(ny, nx)
    qs_int.assign(np.full((ny, nx + 1), seed, dtype=np.float32))
    for _ in range(steps):
        wp.launch(
            accumulate_qs_x,
            dim=(ny, nx - 1),
            inputs=[
                _f32(np.full((ny, nx + 1), q)), _f32(z + depth), _f32(z),
                _f32(np.full((ny, nx), n)), _f32(np.full((ny, nx), d50)), dt, qs_int, qs_comp,
            ],
            device=DEV,
        )  # fmt: skip

    increment = float(capacity_from_flow(q, depth, n, d50)) * dt
    assert increment < 0.5 * np.spacing(np.float32(seed))  # the add is a no-op on its own
    moved = qs_int.numpy()[:, 1:nx].astype(np.float64) - seed
    assert np.allclose(moved, increment * steps, rtol=0.05)

    # Uncompensated, under identical conditions, the accumulator is still exactly 4.
    naive = np.float32(seed)
    for _ in range(steps):
        naive = np.float32(naive + np.float32(increment))
    assert naive == np.float32(seed)

    # ... and the mechanism itself is live, not just its effect.
    assert (qs_comp.numpy()[:, 1:nx] != 0.0).any()


def test_the_two_directions_are_transposes_of_each_other():
    """The y kernels are the x kernels with the axes swapped -- bit for bit.

    ``grid.py`` calls staggered index/offset errors *the* classic shallow-water bug,
    and a gate fixture must be axis-aligned (the face-normal ``theta`` reads low on
    diagonal flow), so y is the direction most likely to be right by inspection and
    never measured. Transposing the whole problem pins the row offset, the face
    indexing and the divergence's sign in both kernels at once: ``i = ii`` instead of
    ``ii + 1``, ``eta[i+1]`` for ``eta[i-1]``, or a flipped ``div_y`` all fail here
    and nowhere else.
    """
    ny, nx, dt, dx, p = 3, 6, 2.5, 50.0, 0.4
    rng = np.random.default_rng(19)
    z = (20.0 - 0.1 * np.arange(nx)[None, :] * np.ones((ny, 1))).astype(np.float32)
    eta = (z + rng.uniform(0.3, 1.5, (ny, nx))).astype(np.float32)
    qx = rng.uniform(-2.5, 2.5, (ny, nx + 1)).astype(np.float32)
    n = rng.uniform(0.025, 0.05, (ny, nx)).astype(np.float32)
    d50 = rng.uniform(2e-3, 2e-2, (ny, nx)).astype(np.float32)

    qs_x, comp_x = _plain_accumulator(ny, nx)
    wp.launch(
        accumulate_qs_x,
        dim=(ny, nx - 1),
        inputs=[_f32(qx), _f32(eta), _f32(z), _f32(n), _f32(d50), dt, qs_x, comp_x],
        device=DEV,
    )
    # The same problem rotated: the grid is (nx, ny) and the flow runs down columns.
    qs_y = wp.zeros((nx + 1, ny), dtype=wp.float32, device=DEV)
    comp_y = wp.zeros((nx + 1, ny), dtype=wp.float32, device=DEV)
    wp.launch(
        accumulate_qs_y,
        dim=(nx - 1, ny),
        inputs=[
            _f32(qx.T), _f32(eta.T), _f32(z.T), _f32(n.T), _f32(d50.T), dt, qs_y, comp_y
        ],
        device=DEV,
    )  # fmt: skip
    assert (qs_y.numpy() == qs_x.numpy().T).all()

    # ... and Exner reads the two directions the same way round.
    dz_x, _ = _exner(qs_x.numpy(), np.zeros((ny + 1, nx)), dx=dx, porosity=p)
    dz_y, _ = _exner(np.zeros((nx, ny + 1)), qs_y.numpy(), dx=dx, porosity=p)
    assert (dz_y == dz_x.T).all()
    assert np.abs(dz_x).max() > 0.0  # the fixture actually has a divergence to get wrong


def test_still_water_writes_no_transport_at_all():
    """A lake at rest has zero shear, so the integral stays bit-exact zero."""
    ny, nx = 4, 4
    z = _flat_channel_bed(ny, nx)
    qs_int, qs_comp = _plain_accumulator(ny, nx)
    qs_int_y = wp.zeros((ny + 1, nx), dtype=wp.float32, device=DEV)
    qs_comp_y = wp.zeros((ny + 1, nx), dtype=wp.float32, device=DEV)
    args = [
        _f32(z + 2.0), _f32(z), _f32(np.full((ny, nx), 0.03)), _f32(np.full((ny, nx), 0.008)), 10.0
    ]  # fmt: skip
    wp.launch(
        accumulate_qs_x,
        dim=(ny, nx - 1),
        inputs=[wp.zeros((ny, nx + 1), dtype=wp.float32, device=DEV), *args, qs_int, qs_comp],
        device=DEV,
    )
    wp.launch(
        accumulate_qs_y,
        dim=(ny - 1, nx),
        inputs=[
            wp.zeros((ny + 1, nx), dtype=wp.float32, device=DEV), *args, qs_int_y, qs_comp_y
        ],
        device=DEV,
    )  # fmt: skip
    assert (qs_int.numpy() == 0.0).all()
    assert (qs_int_y.numpy() == 0.0).all()


def test_a_channel_face_carries_its_conveyance_fraction():
    """The ``dx/w`` bookkeeping, in the one configuration that isolates it (M7 §4).

    With the water surface below bank level the floodplain component is dry, so the
    whole cell-width flux is the channel's, scaled by ``w/dx``. Getting the mix
    wrong in either direction is a factor of ``dx/w`` -- 10x here -- and the channel
    component must be computed from the channel's own flow depth and hydraulic
    radius ``A/P``, not the cell-mean depth.
    """
    ny, nx, dx, dt = 1, 4, 200.0, 1.0
    w, bank_depth, n_ch, d50 = 20.0, 1.0, 0.028, 0.008
    z = _flat_channel_bed(ny, nx)
    h_ch = 0.5  # water surface half a metre above the invert, half a metre below bank
    eta = z - bank_depth + h_ch
    q_ch = 1.8  # per unit *channel* width

    qs_int, qs_comp = _plain_accumulator(ny, nx)
    wp.launch(
        accumulate_qs_x_channels,
        dim=(ny, nx - 1),
        inputs=[
            _f32(np.full((ny, nx + 1), q_ch)),          # channel component
            wp.zeros((ny, nx + 1), dtype=wp.float32, device=DEV),  # floodplain: no flow
            _f32(eta), _f32(z),
            _f32(np.full((ny, nx), 0.05)),              # floodplain roughness (unused: dry)
            _f32(np.full((ny, nx), d50)),
            _f32(np.full((ny, nx), w)), _f32(np.full((ny, nx), bank_depth)),
            _f32(np.full((ny, nx), n_ch)),
            dx, dt, qs_int, qs_comp,
        ],
        device=DEV,
    )  # fmt: skip

    radius = w * h_ch / (w + 2.0 * h_ch)
    qs_channel = capacity_from_flow(q_ch, h_ch, n_ch, d50, radius=radius)
    got = qs_int.numpy()[0, 1]
    assert qs_channel > 0.0
    # per-cell-width flux * dx == per-channel-width flux * w  (nothing else can pass)
    assert got * dx == pytest.approx(qs_channel * w, rel=1e-4)
    # and the cell-mean-depth mistake would be off by more than an order of magnitude
    naive = capacity_from_flow(q_ch, h_ch * w / dx, n_ch, d50)
    assert naive > 10.0 * qs_channel


def test_the_channel_kernel_collapses_onto_the_plain_one_at_zero_width():
    """``w = 0`` is the no-channel case, bit for bit (the M6 storage-curve property)."""
    ny, nx, dx, dt = 3, 6, 50.0, 3.0
    z = _flat_channel_bed(ny, nx, slope_per_cell=0.05)
    rng = np.random.default_rng(3)
    eta = z + rng.uniform(0.3, 1.5, (ny, nx)).astype(np.float32)
    qx = rng.uniform(-2.0, 2.0, (ny, nx + 1)).astype(np.float32)
    n = np.full((ny, nx), 0.033, dtype=np.float32)
    d50 = np.full((ny, nx), 0.006, dtype=np.float32)
    zeros = np.zeros((ny, nx), dtype=np.float32)

    plain, plain_c = _plain_accumulator(ny, nx)
    wp.launch(
        accumulate_qs_x,
        dim=(ny, nx - 1),
        inputs=[_f32(qx), _f32(eta), _f32(z), _f32(n), _f32(d50), dt, plain, plain_c],
        device=DEV,
    )
    chan, chan_c = _plain_accumulator(ny, nx)
    wp.launch(
        accumulate_qs_x_channels,
        dim=(ny, nx - 1),
        inputs=[
            wp.zeros((ny, nx + 1), dtype=wp.float32, device=DEV), _f32(qx),
            _f32(eta), _f32(z), _f32(n), _f32(d50),
            _f32(zeros), _f32(zeros), _f32(zeros),
            dx, dt, chan, chan_c,
        ],
        device=DEV,
    )  # fmt: skip
    assert (plain.numpy() == chan.numpy()).all()


def test_clearing_keeps_the_compensation_debt():
    """An activation consumes the integral; the bits it never held stay owed."""
    ny, nx = 2, 3
    qs_int, qs_comp = _plain_accumulator(ny, nx)
    qs_int_y = wp.zeros((ny + 1, nx), dtype=wp.float32, device=DEV)
    qs_comp.assign(np.full((ny, nx + 1), 1.25e-9, dtype=np.float32))
    qs_int.assign(np.full((ny, nx + 1), 4.0, dtype=np.float32))
    clear_transport_integral(qs_int, qs_int_y)
    assert (qs_int.numpy() == 0.0).all()
    assert (qs_comp.numpy() == 1.25e-9).all()


# --- Exner -------------------------------------------------------------------


def _exner(qs_x, qs_y, *, dx=50.0, porosity=0.4, lo=None, hi=None):
    ny, nx = qs_y.shape[0] - 1, qs_x.shape[1] - 1
    dz = wp.zeros((ny, nx), dtype=wp.float64, device=DEV)
    banked = wp.zeros((ny, nx), dtype=wp.float64, device=DEV)
    lo_a = _f32(np.full((ny, nx), -np.inf) if lo is None else lo)
    hi_a = _f32(np.full((ny, nx), np.inf) if hi is None else hi)
    wp.launch(
        exner_update,
        dim=(ny, nx),
        inputs=[_f32(qs_x), _f32(qs_y), lo_a, hi_a, dx, 1.0 / (1.0 - porosity), dz, banked],
        device=DEV,
    )
    return dz.numpy(), banked.numpy()


def test_uniform_transport_moves_no_bed():
    """A divergence-free bedload field erodes nothing anywhere, bit-exactly."""
    ny, nx = 4, 4
    dz, banked = _exner(np.full((ny, nx + 1), 3.7e-3), np.full((ny + 1, nx), -1.1e-3))
    assert (dz == 0.0).all()
    assert (banked == 0.0).all()


def test_exner_matches_the_hand_computed_divergence():
    """A linear ramp in ``q_s`` erodes every cell by the same hand-computable amount."""
    ny, nx, dx, p = 3, 5, 50.0, 0.4
    gain = 2.0e-3  # m^2 of integrated transport gained per face crossed
    qs_x = gain * np.arange(nx + 1, dtype=np.float64)[None, :] * np.ones((ny, 1))
    dz, _ = _exner(qs_x, np.zeros((ny + 1, nx)), dx=dx, porosity=p)
    expected = -gain / dx / (1.0 - p)  # net export => the bed drops
    assert np.allclose(dz, expected, rtol=1e-12)
    assert expected < 0.0


def test_a_bound_banks_exactly_what_it_refuses():
    """Structure cells and a bedrock floor are one mechanism, and nothing is clamped away.

    Silently clamping the bed would invent or destroy solid mass the way a bare
    ``max(h, 0)`` invents water (M4). The refused metres are accumulated for the
    sediment ledger, which converts them at ``A*(1-p)``.
    """
    ny, nx, dx, p = 1, 3, 50.0, 0.4
    gain = 2.0e-3
    qs_x = gain * np.arange(nx + 1, dtype=np.float64)[None, :] * np.ones((ny, 1))
    want = -gain / dx / (1.0 - p)

    frozen = np.zeros((ny, nx), dtype=np.float32)  # a dam: lo = hi = 0
    dz, banked = _exner(qs_x, np.zeros((ny + 1, nx)), dx=dx, porosity=p, lo=frozen, hi=frozen)
    assert (dz == 0.0).all()
    assert np.allclose(banked, want, rtol=1e-12)

    floor = np.full((ny, nx), 0.5 * want, dtype=np.float32)  # bedrock halfway down
    dz, banked = _exner(qs_x, np.zeros((ny + 1, nx)), dx=dx, porosity=p, lo=floor)
    assert np.allclose(dz, floor, rtol=1e-6)
    assert np.allclose(banked, want - floor, rtol=1e-5)


def test_the_bed_is_rebuilt_from_the_pristine_bed_not_incremented():
    """``z = float32(z0 + dz_cum)``: the increment survives even though ``z += dz`` cannot.

    At ``z ~ 175 m`` float32 has ``eps = 1.5e-5 m``, so a 1e-6 m activation increment
    added in place is discarded in full, every time, forever -- not drift, a deleted
    term (M7 §1.1). Accumulating in float64 and recomputing keeps it.
    """
    ny, nx = 2, 2
    z0 = np.full((ny, nx), 175.0, dtype=np.float32)
    dz = wp.zeros((ny, nx), dtype=wp.float64, device=DEV)
    z = _f32(z0.copy())
    increment = 1.0e-6
    activations = 100
    dz_host = dz.numpy()
    for k in range(activations):
        dz_host[:] = increment * (k + 1)
        dz.assign(dz_host)
        wp.launch(rebuild_bed, dim=(ny, nx), inputs=[_f32(z0), dz, z], device=DEV)
    # The rebuilt bed carries the accumulated change to within one ulp of `z` -- which
    # is the best any float32 bed can do, and is the whole difference from zero below.
    moved = z.numpy() - z0
    assert np.abs(moved - increment * activations).max() <= np.spacing(np.float32(175.0))

    # The same increments added in place at float32 never move the bed at all.
    naive = np.float32(175.0)
    for _ in range(activations):
        naive = np.float32(naive + np.float32(increment))
    assert naive == np.float32(175.0)


# --- celerity ----------------------------------------------------------------


def test_bed_celerity_matches_a_finite_difference_of_the_law():
    """The analytical ``c_b`` is the derivative of the capacity actually implemented.

    ``dq_s/dz = -dq_s/dh`` at fixed unit discharge: a rise in the bed thins the flow,
    which raises the shear and so the capacity. Both the celerity gate and the
    morphological-CFL print are computed from this, so it is checked against a
    central difference of :func:`capacity_from_flow` rather than trusted.
    """
    q, h, n, d50, p = 2.0, 1.0, 0.03, 0.008, 0.4
    eps = 1e-6
    fd = (capacity_from_flow(q, h - eps, n, d50) - capacity_from_flow(q, h + eps, n, d50)) / (
        2 * eps
    )
    assert bed_celerity(q, h, n, d50, p) == pytest.approx(fd / (1 - p), rel=1e-5)

    # ... and with a channel section, where R = A/P varies with h too.
    w = 20.0

    def qs_at(depth: float) -> float:
        return float(capacity_from_flow(q, depth, n, d50, radius=w * depth / (w + 2 * depth)))

    fd_ch = (qs_at(h - eps) - qs_at(h + eps)) / (2 * eps)
    assert bed_celerity(q, h, n, d50, p, width=w) == pytest.approx(fd_ch / (1 - p), rel=1e-5)


def test_a_sub_grid_channels_bed_wave_is_slowed_by_its_conveyance_fraction():
    """The flux is per channel width; the bed change is spread over the whole cell."""
    q, h, n, d50, p, w, dx = 2.0, 1.0, 0.03, 0.008, 0.4, 20.0, 200.0
    full = bed_celerity(q, h, n, d50, p, width=w)
    cell = bed_celerity(q, h, n, d50, p, width=w, dx=dx)
    assert cell == pytest.approx(full * w / dx, rel=1e-12)


def test_the_morphological_courant_number_is_the_ratio_it_claims():
    assert morphological_courant(1e-4, 900.0, 50.0) == pytest.approx(1.8e-3)
    assert morphological_courant(0.0, 900.0, 50.0) == 0.0


def test_no_shear_means_no_celerity():
    """Below threshold the bed wave does not exist, so the CFL print says zero."""
    assert bed_celerity(0.0, 1.0, 0.03, 0.008, 0.4) == 0.0
    deep = 20.0  # so slow that theta < theta_c
    assert shields_from_flow(1.0, deep, 0.03, 0.008) < SHIELDS_CRITICAL
    assert bed_celerity(1.0, deep, 0.03, 0.008, 0.4) == 0.0
