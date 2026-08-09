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

from solver.core import sources
from solver.core.boundaries import apply_closed_bc, apply_open_outflow
from solver.core.channels import (
    channel_flow_depth,
    channel_radius,
    column_depth,
    eta_subgrid,
    face_channel_bed,
    face_channel_width,
    floodplain_flow_depth,
)
from solver.core.friction import manning_denominator, manning_denominator_radius
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
    n: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    dt: wp.float32,
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
    n_face = 0.5 * (n[i, j - 1] + n[i, j])  # uniform n -> 0.5*(n+n)==n (bit-exact)
    slope = (eta_r - eta_l) / dx  # d(h+z)/dx across the face
    num = q - g * h_flow * dt * slope
    den = manning_denominator(q, h_flow, n_face, g, dt)
    qx[i, j] = num / den


@wp.kernel
def update_qy(
    qy: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    n: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    dt: wp.float32,
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
    n_face = 0.5 * (n[i - 1, j] + n[i, j])  # uniform n -> 0.5*(n+n)==n (bit-exact)
    slope = (eta_b - eta_t) / dx
    num = q - g * h_flow * dt * slope
    den = manning_denominator(q, h_flow, n_face, g, dt)
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
def apply_rain_field(
    h: wp.array2d(dtype=wp.float32),
    rain: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    scale: wp.float32,
):
    """Add a spatial rainfall source ``rain[i,j]*scale*dt`` (M3 field rainfall).

    ``scale`` is the temporal on/off multiplier (1 while raining, 0 otherwise);
    the spatial pattern is static so mass inflow stays analytic (§8).
    """
    i, j = wp.tid()
    h[i, j] = h[i, j] + rain[i, j] * scale * dt


@wp.kernel
def apply_infiltration(
    h: wp.array2d(dtype=wp.float32),
    infil: wp.array2d(dtype=wp.float32),
    loss_cum: wp.array2d(dtype=wp.float64),
    dt: wp.float32,
):
    """Remove a capped infiltration loss and bank it in ``loss_cum`` (M3 sink).

    Constant-rate (Horton-final) infiltration: a cell loses ``infil*dt`` but never
    more than it holds, so depth stays non-negative. We bank the **exact** depth
    the float32 field lost (``f64(avail) - f64(h_new)``) into the float64
    accumulator (one writer -> deterministic, §8/§12), so the outflow the ledger
    reads mirrors ``h`` bit-for-bit and the residual stays at field quantization.
    """
    i, j = wp.tid()
    avail = h[i, j]
    inf = infil[i, j] * dt
    if inf > avail:
        inf = avail
    if inf < 0.0:
        inf = 0.0
    h_new = avail - inf
    h[i, j] = h_new
    loss_cum[i, j] = loss_cum[i, j] + (wp.float64(avail) - wp.float64(h_new))


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


# --- M6 sub-grid channels ---------------------------------------------------
# A cell may carry a channel narrower than itself (solver.core.channels). Three
# things change and nothing else does: eta comes from the storage curve, a face
# carries two flows instead of one, and the timestep is set by the water *column*
# rather than the cell-mean depth. Continuity, the limiter's beta, the boundaries
# and the ledger are untouched -- the two components are recombined into exactly
# the same total per-cell-width flux `qx`/`qy` those already speak.


@wp.kernel
def compute_eta_channels(
    h: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    chan_w: wp.array2d(dtype=wp.float32),
    chan_d: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    eta: wp.array2d(dtype=wp.float32),
):
    """``eta`` through the sub-grid storage curve (``w = 0`` -> ``h + z``)."""
    i, j = wp.tid()
    eta[i, j] = eta_subgrid(h[i, j], z[i, j], chan_w[i, j], chan_d[i, j], dx)


@wp.func
def _channel_flux(
    q_prev: wp.float32,
    eta_max: wp.float32,
    slope: wp.float32,
    w_face: wp.float32,
    z_bank: wp.float32,
    z_ch: wp.float32,
    n_ch: wp.float32,
    dt: wp.float32,
    g: wp.float32,
) -> wp.float32:
    """Bates update for the channel component, per unit **channel** width.

    The channel carries the water between the channel bed ``z_ch`` and the bank
    ``z_bank`` (the floodplain bed at the face); anything above that is the
    floodplain component's, so the two never double-count. Hydraulic radius is
    ``A/P``, not the depth -- see :func:`solver.core.friction.manning_denominator_radius`.
    """
    if w_face <= 0.0:
        return 0.0
    h_ch = channel_flow_depth(eta_max, z_bank, z_ch)  # saturates at bank full
    if h_ch < H_DRY:
        return 0.0
    radius = channel_radius(w_face, h_ch)
    num = q_prev - g * h_ch * dt * slope
    den = manning_denominator_radius(q_prev, h_ch, radius, n_ch, g, dt)
    return num / den


@wp.func
def _floodplain_flux(
    q_prev: wp.float32,
    eta_max: wp.float32,
    slope: wp.float32,
    z_bank: wp.float32,
    n_fp: wp.float32,
    dt: wp.float32,
    g: wp.float32,
) -> wp.float32:
    """Bates update for the floodplain component -- the M1 face update verbatim."""
    h_fp = floodplain_flow_depth(eta_max, z_bank)
    if h_fp < H_DRY:
        return 0.0
    num = q_prev - g * h_fp * dt * slope
    den = manning_denominator(q_prev, h_fp, n_fp, g, dt)
    return num / den


@wp.kernel
def update_qx_channels(
    qx: wp.array2d(dtype=wp.float32),
    qx_ch: wp.array2d(dtype=wp.float32),
    qx_fp: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    n: wp.array2d(dtype=wp.float32),
    chan_w: wp.array2d(dtype=wp.float32),
    chan_d: wp.array2d(dtype=wp.float32),
    chan_n: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    dt: wp.float32,
    g: wp.float32,
):
    """Two-component interior x-face update. Launched over ``(ny, nx-1)``."""
    i, jj = wp.tid()
    j = jj + 1

    eta_l = eta[i, j - 1]
    eta_r = eta[i, j]
    eta_max = wp.max(eta_l, eta_r)
    slope = (eta_r - eta_l) / dx
    z_bank = wp.max(z[i, j - 1], z[i, j])

    q_fp = _floodplain_flux(
        qx_fp[i, j], eta_max, slope, z_bank, 0.5 * (n[i, j - 1] + n[i, j]), dt, g
    )
    # A channel conveys only where it is continuous across the face; the narrower
    # section controls, and the higher channel bed is the sill.
    w_face = face_channel_width(chan_w[i, j - 1], chan_w[i, j])
    z_ch = face_channel_bed(z[i, j - 1], chan_d[i, j - 1], z[i, j], chan_d[i, j])
    q_ch = _channel_flux(
        qx_ch[i, j],
        eta_max,
        slope,
        w_face,
        z_bank,
        z_ch,
        0.5 * (chan_n[i, j - 1] + chan_n[i, j]),
        dt,
        g,
    )

    qx_fp[i, j] = q_fp
    qx_ch[i, j] = q_ch
    frac = w_face / dx
    qx[i, j] = q_fp * (1.0 - frac) + q_ch * frac


@wp.kernel
def update_qy_channels(
    qy: wp.array2d(dtype=wp.float32),
    qy_ch: wp.array2d(dtype=wp.float32),
    qy_fp: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    n: wp.array2d(dtype=wp.float32),
    chan_w: wp.array2d(dtype=wp.float32),
    chan_d: wp.array2d(dtype=wp.float32),
    chan_n: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    dt: wp.float32,
    g: wp.float32,
):
    """Two-component interior y-face update. Launched over ``(ny-1, nx)``."""
    ii, j = wp.tid()
    i = ii + 1

    eta_t = eta[i - 1, j]
    eta_b = eta[i, j]
    eta_max = wp.max(eta_t, eta_b)
    slope = (eta_b - eta_t) / dx
    z_bank = wp.max(z[i - 1, j], z[i, j])

    q_fp = _floodplain_flux(
        qy_fp[i, j], eta_max, slope, z_bank, 0.5 * (n[i - 1, j] + n[i, j]), dt, g
    )
    w_face = face_channel_width(chan_w[i - 1, j], chan_w[i, j])
    z_ch = face_channel_bed(z[i - 1, j], chan_d[i - 1, j], z[i, j], chan_d[i, j])
    q_ch = _channel_flux(
        qy_ch[i, j],
        eta_max,
        slope,
        w_face,
        z_bank,
        z_ch,
        0.5 * (chan_n[i - 1, j] + chan_n[i, j]),
        dt,
        g,
    )

    qy_fp[i, j] = q_fp
    qy_ch[i, j] = q_ch
    frac = w_face / dx
    qy[i, j] = q_fp * (1.0 - frac) + q_ch * frac


@wp.kernel
def limit_qx_channels(
    qx: wp.array2d(dtype=wp.float32),
    qx_ch: wp.array2d(dtype=wp.float32),
    qx_fp: wp.array2d(dtype=wp.float32),
    beta: wp.array2d(dtype=wp.float32),
):
    """Scale an x-face and *both* its components by the total flux's donor beta.

    The donor is chosen from the **total** flux, which is what continuity moves and
    what ``beta`` was computed against; scaling both components by the same factor
    keeps ``q = q_fp·(1-frac) + q_ch·frac`` exact, so the M1 guarantee (one face,
    scaled once, by its donor cell) survives unchanged.
    """
    i, jj = wp.tid()
    j = jj + 1
    q = qx[i, j]
    if q == 0.0:
        return
    b = beta[i, j - 1]
    if q < 0.0:
        b = beta[i, j]
    qx[i, j] = q * b
    qx_ch[i, j] = qx_ch[i, j] * b
    qx_fp[i, j] = qx_fp[i, j] * b


@wp.kernel
def limit_qy_channels(
    qy: wp.array2d(dtype=wp.float32),
    qy_ch: wp.array2d(dtype=wp.float32),
    qy_fp: wp.array2d(dtype=wp.float32),
    beta: wp.array2d(dtype=wp.float32),
):
    """Scale a y-face and both its components by the total flux's donor beta."""
    ii, j = wp.tid()
    i = ii + 1
    q = qy[i, j]
    if q == 0.0:
        return
    b = beta[i - 1, j]
    if q < 0.0:
        b = beta[i, j]
    qy[i, j] = q * b
    qy_ch[i, j] = qy_ch[i, j] * b
    qy_fp[i, j] = qy_fp[i, j] * b


@wp.kernel
def reduce_column_max(
    h: wp.array2d(dtype=wp.float32),
    chan_w: wp.array2d(dtype=wp.float32),
    chan_d: wp.array2d(dtype=wp.float32),
    dx: wp.float32,
    out_max: wp.array(dtype=wp.float32),
):
    """Atomic-max of the water *column* depth -- the wave speed a channel really has.

    ``h`` is a cell mean; a channel concentrates it by ``dx/w``, so reducing over
    ``h`` would under-resolve the timestep by exactly that factor (a 0.1 m mean over
    a 20 m channel in a 200 m cell is a **1 m** column). Without channels this is
    identical to :func:`reduce_hmax`.
    """
    i, j = wp.tid()
    wp.atomic_max(out_max, 0, column_depth(h[i, j], chan_w[i, j], chan_d[i, j], dx))


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

    With sub-grid channels armed (M6) the reduction is over the water **column**
    (:func:`reduce_column_max`) rather than the cell mean, since that is the depth
    the wave actually sees.
    """
    state.h_max.zero_()
    chan = state.channels
    if chan is not None:
        wp.launch(
            reduce_column_max,
            dim=state.grid.shape,
            inputs=[state.h, chan.w, chan.d, float(state.grid.dx), state.h_max],
            device=state.device,
        )
    else:
        wp.launch(
            reduce_hmax, dim=state.grid.shape, inputs=[state.h, state.h_max], device=state.device
        )
    h_max = float(state.h_max.numpy()[0])
    if h_max <= H_DRY:
        return dt_max
    dt = alpha * state.grid.dx / math.sqrt(GRAVITY * h_max)
    return min(dt, dt_max)


def step(
    state: State,
    dt: float,
    rain: float = 0.0,
    limit: bool = True,
    rain_scale: float = 1.0,
    t: float = 0.0,  # noqa: ARG001 -- scheme-interface parity; LI has no time-dependent BC
) -> None:
    """Advance the state by one local-inertial step of size ``dt`` (seconds).

    Order: refresh ``eta`` -> update x/y face discharges (friction folded in, from
    the per-cell roughness field ``state.n``) -> re-assert closed boundaries ->
    (optional) mass-conservative outflow limiter -> continuity + uniform rainfall
    -> spatial rain field (if any) -> infiltration sink (if any). Rain is a
    velocity (m/s): ``rate_mm_hr / 1000 / 3600``.

    ``rain`` is a *uniform* scalar source (M1/M2 path); ``state.rain`` carries the
    optional M3 spatial rain field, temporally gated by ``rain_scale`` (1 while
    raining, 0 otherwise). ``state.infil`` is the optional M3 infiltration sink.
    Both are skipped (no launch, bitwise no-op) when unset -- so uniform-parameter
    runs such as dam-break are unchanged.

    ``t`` (the simulated time at the start of the step) is accepted for parity with
    the scheme interface (:mod:`solver.core.schemes`) and ignored: the local-inertial
    scheme has no time-dependent boundary forcing -- ``fixed_stage`` is HLLC-only
    (M5 plan §1.4) -- and its rainfall gating is done by the caller.

    ``state.channels`` (M6, :mod:`solver.core.channels`) switches the eta, face and
    limiter launches onto their sub-grid counterparts: ``eta`` comes from the
    storage curve and each face carries a channel flow plus a floodplain flow,
    recombined into the same total ``qx``/``qy`` continuity already reads. Unarmed,
    none of that code is launched, so pre-M6 runs are bitwise-identical.

    ``limit`` enables the per-cell donor limiter that keeps depths non-negative
    when the scheme is pushed out of regime (steep thin-sheet flow). It is
    inactive (``beta == 1``) whenever no cell is over-drained, so it does not
    perturb in-regime runs such as the dam-break validation.
    """
    g = state.grid
    dxf, dtf, gf = float(g.dx), float(dt), float(GRAVITY)
    chan = state.channels  # M6 sub-grid channels; None -> the M1 kernels, untouched

    if chan is None:
        wp.launch(
            compute_eta, dim=g.shape, inputs=[state.h, state.z, state.eta], device=state.device
        )
        if g.nx > 1:
            wp.launch(
                update_qx,
                dim=(g.ny, g.nx - 1),
                inputs=[state.qx, state.eta, state.z, state.n, dxf, dtf, gf],
                device=state.device,
            )
        if g.ny > 1:
            wp.launch(
                update_qy,
                dim=(g.ny - 1, g.nx),
                inputs=[state.qy, state.eta, state.z, state.n, dxf, dtf, gf],
                device=state.device,
            )
    else:
        wp.launch(
            compute_eta_channels,
            dim=g.shape,
            inputs=[state.h, state.z, chan.w, chan.d, dxf, state.eta],
            device=state.device,
        )
        if g.nx > 1:
            wp.launch(
                update_qx_channels,
                dim=(g.ny, g.nx - 1),
                inputs=[
                    state.qx, chan.qx_ch, chan.qx_fp, state.eta, state.z, state.n,
                    chan.w, chan.d, chan.n, dxf, dtf, gf,
                ],
                device=state.device,
            )  # fmt: skip
        if g.ny > 1:
            wp.launch(
                update_qy_channels,
                dim=(g.ny - 1, g.nx),
                inputs=[
                    state.qy, chan.qy_ch, chan.qy_fp, state.eta, state.z, state.n,
                    chan.w, chan.d, chan.n, dxf, dtf, gf,
                ],
                device=state.device,
            )  # fmt: skip
    apply_closed_bc(state)

    if limit:
        wp.launch(
            compute_outflow_beta,
            dim=g.shape,
            inputs=[state.h, state.qx, state.qy, dxf, dtf, state.beta],
            device=state.device,
        )
        if g.nx > 1:
            if chan is None:
                wp.launch(
                    limit_qx,
                    dim=(g.ny, g.nx - 1),
                    inputs=[state.qx, state.beta],
                    device=state.device,
                )
            else:
                wp.launch(
                    limit_qx_channels,
                    dim=(g.ny, g.nx - 1),
                    inputs=[state.qx, chan.qx_ch, chan.qx_fp, state.beta],
                    device=state.device,
                )
        if g.ny > 1:
            if chan is None:
                wp.launch(
                    limit_qy,
                    dim=(g.ny - 1, g.nx),
                    inputs=[state.qy, state.beta],
                    device=state.device,
                )
            else:
                wp.launch(
                    limit_qy_channels,
                    dim=(g.ny - 1, g.nx),
                    inputs=[state.qy, chan.qy_ch, chan.qy_fp, state.beta],
                    device=state.device,
                )

    # Areal sources: compensated when armed (solver.core.sources). Continuity then
    # carries no rain of its own -- the source add belongs to the Kahan kernel, which
    # needs it as a separate rounding to compensate. Unarmed, both stay fused/plain
    # and the arithmetic is untouched.
    compensated = state.h_comp is not None
    wp.launch(
        update_h,
        dim=g.shape,
        inputs=[state.h, state.qx, state.qy, dxf, dtf, 0.0 if compensated else float(rain)],
        device=state.device,
    )
    # Exactly one compensated source kernel per step, and it keeps launching after
    # the storm stops (a zero increment) so the outstanding compensation -- up to
    # half an ulp of `h` per cell, which over a reach-scale grid is itself the order
    # of the drift being removed -- is repaid into the field rather than stranded.
    # `state.rain is not None` is exactly field-rain mode: run.py never combines the
    # two, so uniform and field never both fire.
    if compensated and state.rain is None:
        sources.apply_uniform_rain(state, rain, dt)

    if state.rain is not None:
        if compensated:
            sources.apply_rain_field(state, dt, rain_scale)
        else:
            wp.launch(
                apply_rain_field,
                dim=g.shape,
                inputs=[state.h, state.rain, dtf, float(rain_scale)],
                device=state.device,
            )
    if state.infil is not None:
        wp.launch(
            apply_infiltration,
            dim=g.shape,
            inputs=[state.h, state.infil, state.loss_cum, dtf],
            device=state.device,
        )
    if state.open_edges:
        apply_open_outflow(state, dt)
