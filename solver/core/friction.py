"""Manning bed friction for the local-inertial scheme (M1, HANDOFF §8).

The local-inertial flux update folds friction into the flux denominator
(semi-implicit in the *previous* discharge ``q^n``, HANDOFF §8)::

    q^{n+1} = ( q^n - g * h_flow * dt * d(h+z)/dx ) / D
    with    D = 1 + g * dt * n^2 * |q^n| / h_flow^(7/3)

This module exposes ``manning_denominator`` as a Warp ``@wp.func`` so both the
x- and y-face kernels share exactly one definition of ``D``.
"""

from __future__ import annotations

import warp as wp


@wp.func
def manning_denominator(
    q_prev: wp.float32,
    h_flow: wp.float32,
    manning_n: wp.float32,
    g: wp.float32,
    dt: wp.float32,
) -> wp.float32:
    """Semi-implicit Manning denominator ``D`` for a single face.

    ``h_flow`` is assumed already guarded ``>= H_DRY`` by the caller, so the
    ``h_flow^(7/3)`` divisor is safe.
    """
    return 1.0 + g * dt * manning_n * manning_n * wp.abs(q_prev) / wp.pow(h_flow, 7.0 / 3.0)


@wp.func
def manning_denominator_radius(
    q_prev: wp.float32,
    h_flow: wp.float32,
    radius: wp.float32,
    manning_n: wp.float32,
    g: wp.float32,
    dt: wp.float32,
) -> wp.float32:
    """Manning denominator for a face whose hydraulic radius is *not* its depth.

    A sub-grid channel (M6, :mod:`solver.core.channels`) is not wide-and-shallow:
    its wetted perimeter includes two banks, so ``R = A/P = w·h/(w + 2h)`` rather
    than ``R ≈ h``. Writing the friction term per unit channel width,

        D = 1 + g·dt·n²·|q| / (h_flow · R^(4/3))

    which is exactly :func:`manning_denominator` when ``R = h_flow`` (the wide-channel
    limit), so the floodplain and channel forms are one formula with one difference
    that is *physically* the point: at ``w = 5·h`` the radius costs ~25% of the
    conveyance, and ignoring it would bias a coarse run toward too much flow.
    """
    return 1.0 + g * dt * manning_n * manning_n * wp.abs(q_prev) / (
        h_flow * wp.pow(radius, 4.0 / 3.0)
    )
