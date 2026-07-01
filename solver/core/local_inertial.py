"""Local-inertial shallow-water scheme (M1, HANDOFF §8).

Bates, Horritt & Fewtrell (2010): drop the advective-acceleration term from the
momentum equation, leaving an explicit, staggered-grid update that is cheap and
GPU-perfect and remains the permanent *coverage* scheme for lowland floodplains.

The grid layout and sign convention are defined once in :mod:`solver.core.grid`
(read that docstring before touching indices here). In brief:
  * ``h``, ``z``, ``eta = h + z`` are cell-centred ``(ny, nx)``
  * ``qx`` on x-faces ``(ny, nx+1)``, positive flows +x (increasing column)
  * ``qy`` on y-faces ``(ny+1, nx)``, positive flows +y (increasing row)

Per-face momentum update (x-face between cells ``(i, j-1)`` and ``(i, j)``)::

    h_flow = max(eta_L, eta_R) - max(z_L, z_R)          # depth available at face
    if h_flow < H_DRY: q = 0                             # wet/dry guard (no NaN)
    q^{n+1} = ( q^n - g*h_flow*dt*(eta_R - eta_L)/dx ) / D

with ``D`` the Manning denominator (:mod:`solver.core.friction`). Cell continuity
then sums the four bounding face fluxes plus the rainfall source.

Boundary faces (columns 0/nx of ``qx``, rows 0/ny of ``qy``) are never touched by
these kernels -- they stay at their initialised value of 0, which *is* the closed
(reflective) boundary condition. :mod:`solver.core.boundaries` re-asserts this
explicitly for clarity and for future non-closed BC types.
"""

from __future__ import annotations

import math

import warp as wp

from solver.core.boundaries import apply_closed_bc
from solver.core.friction import manning_denominator
from solver.core.grid import GRAVITY, H_DRY
from solver.core.state import State


@wp.func
def face_h_flow(
    eta_a: wp.float32,
    eta_b: wp.float32,
    z_a: wp.float32,
    z_b: wp.float32,
) -> wp.float32:
    """Flow depth at a face: max water-surface minus max bed of the two cells."""
    return wp.max(eta_a, eta_b) - wp.max(z_a, z_b)


@wp.kernel
def compute_eta(
    h: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
):
    """Water-surface elevation ``eta = h + z`` at every cell centre."""
    i, j = wp.tid()
    eta[i, j] = h[i, j] + z[i, j]


@wp.kernel
def update_qx(
    qx: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    dt: wp.float32,
    manning_n: wp.float32,
    g: wp.float32,
):
    """Update interior x-faces. Launched over ``(ny, nx-1)``; ``j = jj + 1``."""
    i, jj = wp.tid()
    j = jj + 1  # interior face index in [1, nx-1]; boundary faces 0 and nx untouched

    eta_l = eta[i, j - 1]
    eta_r = eta[i, j]
    h_flow = face_h_flow(eta_l, eta_r, z[i, j - 1], z[i, j])

    if h_flow < H_DRY:
        qx[i, j] = 0.0
        return

    q = qx[i, j]
    slope = (eta_r - eta_l) / dx  # d(h+z)/dx across the face
    num = q - g * h_flow * dt * slope
    den = manning_denominator(q, h_flow, manning_n, g, dt)
    qx[i, j] = num / den


@wp.kernel
def update_qy(
    qy: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    dt: wp.float32,
    manning_n: wp.float32,
    g: wp.float32,
):
    """Update interior y-faces. Launched over ``(ny-1, nx)``; ``i = ii + 1``."""
    ii, j = wp.tid()
    i = ii + 1  # interior face index in [1, ny-1]; boundary faces 0 and ny untouched

    eta_t = eta[i - 1, j]
    eta_b = eta[i, j]
    h_flow = face_h_flow(eta_t, eta_b, z[i - 1, j], z[i, j])

    if h_flow < H_DRY:
        qy[i, j] = 0.0
        return

    q = qy[i, j]
    slope = (eta_b - eta_t) / dx
    num = q - g * h_flow * dt * slope
    den = manning_denominator(q, h_flow, manning_n, g, dt)
    qy[i, j] = num / den


@wp.kernel
def update_h(
    h: wp.array2d(dtype=wp.float32),
    qx: wp.array2d(dtype=wp.float32),
    qy: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    dt: wp.float32,
    rain: wp.float32,
):
    """Continuity + rainfall source. Net inflow through the four bounding faces.

    ``h[i,j] += dt/dx * (qx[i,j] - qx[i,j+1] + qy[i,j] - qy[i+1,j]) + rain*dt``

    No depth clamp: the wet/dry flux guard in the momentum kernels keeps cells
    from over-draining, and leaving continuity as a pure flux divergence is what
    makes mass conservation exact to float round-off (HANDOFF §8).
    """
    i, j = wp.tid()
    net = qx[i, j] - qx[i, j + 1] + qy[i, j] - qy[i + 1, j]
    h[i, j] = h[i, j] + dt / dx * net + rain * dt


@wp.kernel
def compute_outflow_beta(
    h: wp.array2d(dtype=wp.float32),
    qx: wp.array2d(dtype=wp.float32),
    qy: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    dt: wp.float32,
    beta: wp.array2d(dtype=wp.float32),
):
    """Per-cell outflow-limiter factor ``beta in [0, 1]`` (mass-conservative).

    A cell may not release more water than it holds. Summing the magnitudes of
    its *outgoing* face fluxes gives the requested outflow depth ``dt/dx*Q_out``;
    ``beta = min(1, h / (dt/dx*Q_out))`` scales every outgoing face of this cell
    down so the drained depth is at most ``h``. Because each face is scaled by its
    single *donor* cell (see :func:`limit_qx` / :func:`limit_qy`), the shared face
    value stays consistent for both neighbours and mass is conserved exactly.

    Without this, local-inertial driven out of regime (thin sheets on steep beds)
    overdraws cells to large negative depths and blows up (HANDOFF §8, §12).
    """
    i, j = wp.tid()
    q_out = wp.float32(0.0)
    # left face (outgoing when flux is -x), right face (outgoing when +x)
    ql = qx[i, j]
    if ql < 0.0:
        q_out = q_out - ql
    qr = qx[i, j + 1]
    if qr > 0.0:
        q_out = q_out + qr
    # top face (outgoing when -y), bottom face (outgoing when +y)
    qt = qy[i, j]
    if qt < 0.0:
        q_out = q_out - qt
    qb = qy[i + 1, j]
    if qb > 0.0:
        q_out = q_out + qb

    out_depth = dt / dx * q_out
    if out_depth > 0.0:
        beta[i, j] = wp.clamp(h[i, j] / out_depth, 0.0, 1.0)
    else:
        beta[i, j] = 1.0


@wp.kernel
def limit_qx(qx: wp.array2d(dtype=wp.float32), beta: wp.array2d(dtype=wp.float32)):
    """Scale each interior x-face by its donor (upwind) cell's beta. ``j = jj+1``."""
    i, jj = wp.tid()
    j = jj + 1
    q = qx[i, j]
    if q > 0.0:  # flows +x: donor is the left cell (i, j-1)
        qx[i, j] = q * beta[i, j - 1]
    elif q < 0.0:  # flows -x: donor is the right cell (i, j)
        qx[i, j] = q * beta[i, j]


@wp.kernel
def limit_qy(qy: wp.array2d(dtype=wp.float32), beta: wp.array2d(dtype=wp.float32)):
    """Scale each interior y-face by its donor (upwind) cell's beta. ``i = ii+1``."""
    ii, j = wp.tid()
    i = ii + 1
    q = qy[i, j]
    if q > 0.0:  # flows +y: donor is the top cell (i-1, j)
        qy[i, j] = q * beta[i - 1, j]
    elif q < 0.0:  # flows -y: donor is the bottom cell (i, j)
        qy[i, j] = q * beta[i, j]


@wp.kernel
def reduce_hmax(h: wp.array2d(dtype=wp.float32), out_max: wp.array(dtype=wp.float32)):
    """Atomic-max of the depth field into ``out_max[0]``.

    Max is order-independent, so the atomic stays deterministic across launches
    (HANDOFF §8, §12) -- unlike a float sum, which is why mass accounting is done
    host-side instead (see :mod:`solver.core.massbalance`).
    """
    i, j = wp.tid()
    wp.atomic_max(out_max, 0, h[i, j])


def compute_dt(state: State, alpha: float = 0.7, dt_max: float = 30.0) -> float:
    """Deterministic adaptive timestep from **state**, never wall-clock (§8, §12).

    ``dt = alpha * dx / sqrt(g * h_max)`` (Bates 2010 stability bound), clamped to
    ``dt_max``. ``h_max`` comes from the atomic-max reduction. When the domain is
    effectively dry (``h_max <= H_DRY``) there is nothing to move, so ``dt_max`` is
    returned. The result depends only on field values, so runs reproduce exactly.
    """
    state.h_max.zero_()
    wp.launch(reduce_hmax, dim=state.grid.shape, inputs=[state.h, state.h_max], device=state.device)
    h_max = float(state.h_max.numpy()[0])
    if h_max <= H_DRY:
        return dt_max
    dt = alpha * state.grid.dx / math.sqrt(GRAVITY * h_max)
    return min(dt, dt_max)


def step(state: State, dt: float, manning_n: float, rain: float = 0.0, limit: bool = True) -> None:
    """Advance the state by one local-inertial step of size ``dt`` (seconds).

    Order: refresh ``eta`` -> update x/y face discharges (friction folded in) ->
    re-assert closed boundaries -> (optional) mass-conservative outflow limiter ->
    continuity + rainfall. Rain is a velocity (m/s): ``rate_mm_hr / 1000 / 3600``.

    ``limit`` enables the per-cell donor limiter that keeps depths non-negative
    when the scheme is pushed out of regime (steep thin-sheet flow). It is
    inactive (``beta == 1``) whenever no cell is over-drained, so it does not
    perturb in-regime runs such as the dam-break validation.
    """
    g = state.grid
    dxf, dtf, nf, gf = float(g.dx), float(dt), float(manning_n), float(GRAVITY)

    wp.launch(compute_eta, dim=g.shape, inputs=[state.h, state.z, state.eta], device=state.device)
    if g.nx > 1:
        wp.launch(
            update_qx,
            dim=(g.ny, g.nx - 1),
            inputs=[state.qx, state.eta, state.z, dxf, dtf, nf, gf],
            device=state.device,
        )
    if g.ny > 1:
        wp.launch(
            update_qy,
            dim=(g.ny - 1, g.nx),
            inputs=[state.qy, state.eta, state.z, dxf, dtf, nf, gf],
            device=state.device,
        )
    apply_closed_bc(state)

    if limit:
        wp.launch(
            compute_outflow_beta,
            dim=g.shape,
            inputs=[state.h, state.qx, state.qy, dxf, dtf, state.beta],
            device=state.device,
        )
        if g.nx > 1:
            wp.launch(
                limit_qx, dim=(g.ny, g.nx - 1), inputs=[state.qx, state.beta], device=state.device
            )
        if g.ny > 1:
            wp.launch(
                limit_qy, dim=(g.ny - 1, g.nx), inputs=[state.qy, state.beta], device=state.device
            )

    wp.launch(
        update_h,
        dim=g.shape,
        inputs=[state.h, state.qx, state.qy, dxf, dtf, float(rain)],
        device=state.device,
    )
