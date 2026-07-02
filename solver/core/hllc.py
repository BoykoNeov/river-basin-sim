"""Well-balanced HLLC finite-volume scheme (M4, HANDOFF §8 -- the fidelity scheme).

A cell-centred Godunov finite-volume shallow-water solver: MUSCL slope-limited
reconstruction, Audusse (2004) hydrostatic reconstruction of the bed-slope source
(so a lake at rest is preserved to machine precision), an HLLC approximate Riemann
flux (Toro) at faces, and SSP-RK2 (Heun) time integration with semi-implicit
Manning friction. It is the **fidelity** option for shocks, transcritical flow and
well-balanced wet/dry fronts; the M1 local-inertial scheme remains the permanent
*coverage* scheme (HANDOFF §2). The two coexist behind the scheme-dispatch seam
(:mod:`solver.core.schemes`) -- the run loop calls the same ``compute_dt``/``step``
pair on either.

**2D via an unsplit flux sum, not dimensional splitting.** The plan proposed
dimensional splitting as a first cut; this implementation instead computes the
x-face and y-face HLLC fluxes from the *same* state, sums the two flux divergences
plus a per-direction centered bed-slope source, and wraps the whole ``L(U)`` in one
SSP-RK2. This is cleaner in Warp (a single spatial operator, no intermediate
sweep state), carries no operator-splitting error, and reuses the exact per-face
logic validated in 1-D (:mod:`validation.hllc1d`). Conservative state is
``U = [h, hu, hv]`` (cell-centred); the transverse momentum rides the mass flux,
upwinded on the contact wave, so it is carried correctly through each face.

**Well-balanced reconstruction (the subtlety):** the *static bed* ``z`` is
reconstructed from geometry (a flow-independent slope); the flow is reconstructed
on the water surface ``eta = h + z`` and velocity, and the face depth is
``eta_face - z_face``. Reconstructing the surface (not the depth) makes lake-at-rest
exact; reconstructing ``z`` from geometry (not as ``eta - h``) keeps the centered
bed-slope source correct *in motion*, not only at rest.

Boundaries are transmissive (zero-gradient) here via index clamping in the flux
kernels -- the natural FV free-outflow. Closed/reflective and ``fixed_stage``
ghost-cell BCs are layered on in a later step. Determinism (HANDOFF §8/§12): every
kernel writes each output from one thread reading only inputs; the timestep is a
state-derived atomic-max reduction, order-independent like the LI scheme's.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from solver.core.friction import manning_denominator
from solver.core.grid import GRAVITY, H_DRY
from solver.core.state import State


# --------------------------------------------------------------------------- #
# Scratch buffers (allocated once per run by arm_hllc, attached to state.hllc). #
# --------------------------------------------------------------------------- #
@dataclass
class _HllcScratch:
    """Per-run scratch for the HLLC scheme: primitives, face fluxes, RK buffers."""

    uvel: wp.array  # (ny, nx) cell-centred x-velocity
    vvel: wp.array  # (ny, nx) cell-centred y-velocity
    # x-face fluxes (ny, nx+1): mass, normal-momentum (with L/R bed corrections),
    # transverse-momentum.
    fx_h: wp.array
    fx_mn_l: wp.array
    fx_mn_r: wp.array
    fx_mt: wp.array
    # y-face fluxes (ny+1, nx).
    fy_h: wp.array
    fy_mn_l: wp.array
    fy_mn_r: wp.array
    fy_mt: wp.array
    # spatial-operator output dU/dt (ny, nx)
    dh: wp.array
    dhu: wp.array
    dhv: wp.array
    # RK2 stage-1 state (ny, nx)
    h1: wp.array
    hu1: wp.array
    hv1: wp.array


def arm_hllc(state: State) -> None:
    """Allocate the HLLC momentum fields and scratch on ``state`` (idempotent).

    Momentum starts at zero (fluid at rest -- mirrors the LI ``qx/qy`` zero init),
    so the initial condition is set entirely by ``state.h``.
    """
    if state.hu is not None:
        return
    g = state.grid
    z2 = lambda shape: wp.zeros(shape, dtype=wp.float32, device=state.device)  # noqa: E731
    state.hu = z2(g.shape)
    state.hv = z2(g.shape)
    state.hllc = _HllcScratch(
        uvel=z2(g.shape),
        vvel=z2(g.shape),
        fx_h=z2(g.qx_shape),
        fx_mn_l=z2(g.qx_shape),
        fx_mn_r=z2(g.qx_shape),
        fx_mt=z2(g.qx_shape),
        fy_h=z2(g.qy_shape),
        fy_mn_l=z2(g.qy_shape),
        fy_mn_r=z2(g.qy_shape),
        fy_mt=z2(g.qy_shape),
        dh=z2(g.shape),
        dhu=z2(g.shape),
        dhv=z2(g.shape),
        h1=z2(g.shape),
        hu1=z2(g.shape),
        hv1=z2(g.shape),
    )


# --------------------------------------------------------------------------- #
# Device helpers.                                                              #
# --------------------------------------------------------------------------- #
@wp.func
def _minmod(a: wp.float32, b: wp.float32) -> wp.float32:
    """Minmod limiter: 0 unless a, b share a sign, else the one of smaller |.|."""
    if a * b <= 0.0:
        return wp.float32(0.0)
    if wp.abs(a) < wp.abs(b):
        return a
    return b


@wp.func
def _cget(a: wp.array2d(dtype=wp.float32), i: wp.int32, j: wp.int32, ny: wp.int32, nx: wp.int32):
    """Read a cell value with edge-clamped indices (transmissive ghost cells)."""
    return a[wp.clamp(i, 0, ny - 1), wp.clamp(j, 0, nx - 1)]


@wp.func
def _dryfactor(h0: wp.float32, hm: wp.float32, hp: wp.float32, dry: wp.float32) -> wp.float32:
    """1.0 if a cell and both its stencil neighbours are wet, else 0.0.

    Multiplying a MUSCL slope by this factor drops the reconstruction to
    first-order (piecewise constant) for any cell adjacent to a dry cell. That
    kills the spurious water/bed slope a dry neighbour injects into the minmod
    stencil at a wet/dry front, while preserving well-balancedness: first-order
    Audusse is exactly balanced, so zeroing the slope on both faces of a
    shoreline cell -- consistently in the flux *and* source kernels -- cannot
    break lake-at-rest. Fully-wet interiors are untouched (factor is 1.0).
    """
    if h0 <= dry or hm <= dry or hp <= dry:
        return wp.float32(0.0)
    return wp.float32(1.0)


@wp.func
def _hllc(
    hl: wp.float32,
    ul: wp.float32,
    vtl: wp.float32,
    hr: wp.float32,
    ur: wp.float32,
    vtr: wp.float32,
    g: wp.float32,
    dry: wp.float32,
) -> wp.vec3:
    """HLLC flux ``(Fh, F_normal_mom, F_transverse_mom)`` for one face.

    ``u`` is the face-normal velocity, ``vt`` the transverse velocity (passive). h,
    normal momentum follow the HLL flux in the star region; transverse momentum
    rides the mass flux upwinded on the contact speed. See :mod:`validation.hllc1d`.
    """
    cl = wp.sqrt(g * wp.max(hl, 0.0))
    cr = wp.sqrt(g * wp.max(hr, 0.0))
    dry_l = hl <= dry
    dry_r = hr <= dry
    if dry_l and dry_r:
        return wp.vec3(0.0, 0.0, 0.0)

    # Wave-speed estimates (Toro two-rarefaction; explicit dry branches).
    if dry_l:
        sl = ur - 2.0 * cr
        sr = ur + cr
    elif dry_r:
        sl = ul - cl
        sr = ul + 2.0 * cl
    else:
        hs = 0.5 * (cl + cr) + 0.25 * (ul - ur)
        h_star = wp.max(hs * hs / g, 0.0)
        c_star = wp.sqrt(g * h_star)
        u_star = 0.5 * (ul + ur) + cl - cr
        sl = wp.min(ul - cl, u_star - c_star)
        sr = wp.max(ur + cr, u_star + c_star)

    fh_l = hl * ul
    fmn_l = hl * ul * ul + 0.5 * g * hl * hl
    fh_r = hr * ur
    fmn_r = hr * ur * ur + 0.5 * g * hr * hr

    fh = wp.float32(0.0)
    fmn = wp.float32(0.0)
    if sl >= 0.0:
        fh = fh_l
        fmn = fmn_l
    elif sr <= 0.0:
        fh = fh_r
        fmn = fmn_r
    else:
        inv = 1.0 / (sr - sl)
        fh = (sr * fh_l - sl * fh_r + sl * sr * (hr - hl)) * inv
        fmn = (sr * fmn_l - sl * fmn_r + sl * sr * (hr * ur - hl * ul)) * inv

    # Contact wave speed selects the upwind transverse velocity.
    den_s = hr * (ur - sr) - hl * (ul - sl)
    s_star = 0.5 * (ul + ur)
    if wp.abs(den_s) > 1.0e-12:
        s_star = (sl * hr * (ur - sr) - sr * hl * (ul - sl)) / den_s
    vt = vtr
    if s_star >= 0.0:
        vt = vtl
    return wp.vec3(fh, fmn, fh * vt)


# --------------------------------------------------------------------------- #
# Kernels: primitives, face fluxes, spatial operator, RK stages, friction.    #
# --------------------------------------------------------------------------- #
@wp.kernel
def _primitives(
    h: wp.array2d(dtype=wp.float32),
    hu: wp.array2d(dtype=wp.float32),
    hv: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    uvel: wp.array2d(dtype=wp.float32),
    vvel: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    dry: wp.float32,
):
    """Cell velocities (guarded at dry) and water-surface ``eta = h + z``."""
    i, j = wp.tid()
    hh = h[i, j]
    if hh > dry:
        uvel[i, j] = hu[i, j] / hh
        vvel[i, j] = hv[i, j] / hh
    else:
        uvel[i, j] = 0.0
        vvel[i, j] = 0.0
    eta[i, j] = hh + z[i, j]


@wp.kernel
def _flux_x(
    h: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    uvel: wp.array2d(dtype=wp.float32),
    vvel: wp.array2d(dtype=wp.float32),
    fx_h: wp.array2d(dtype=wp.float32),
    fx_mn_l: wp.array2d(dtype=wp.float32),
    fx_mn_r: wp.array2d(dtype=wp.float32),
    fx_mt: wp.array2d(dtype=wp.float32),
    ny: wp.int32,
    nx: wp.int32,
    g: wp.float32,
    dry: wp.float32,
):
    """HLLC flux on x-face ``j`` (between cells ``(i, j-1)`` and ``(i, j)``).

    Launched over all faces ``(ny, nx+1)``; boundary faces use edge-clamped
    neighbours (transmissive). Normal velocity is ``u``, transverse is ``v``.
    """
    i, j = wp.tid()
    # Stencil columns j-2, j-1, j, j+1 (clamped) for the left/right cell slopes.
    e0 = _cget(eta, i, j - 2, ny, nx)
    e1 = _cget(eta, i, j - 1, ny, nx)
    e2 = _cget(eta, i, j, ny, nx)
    e3 = _cget(eta, i, j + 1, ny, nx)
    z0 = _cget(z, i, j - 2, ny, nx)
    z1 = _cget(z, i, j - 1, ny, nx)
    z2 = _cget(z, i, j, ny, nx)
    z3 = _cget(z, i, j + 1, ny, nx)
    u0 = _cget(uvel, i, j - 2, ny, nx)
    u1 = _cget(uvel, i, j - 1, ny, nx)
    u2 = _cget(uvel, i, j, ny, nx)
    u3 = _cget(uvel, i, j + 1, ny, nx)
    v0 = _cget(vvel, i, j - 2, ny, nx)
    v1 = _cget(vvel, i, j - 1, ny, nx)
    v2 = _cget(vvel, i, j, ny, nx)
    v3 = _cget(vvel, i, j + 1, ny, nx)

    # Drop to first-order for any cell adjacent to a dry cell (see _dryfactor):
    # kills the spurious wet/bed slope a dry neighbour injects at the shoreline.
    h0 = _cget(h, i, j - 2, ny, nx)
    h1 = _cget(h, i, j - 1, ny, nx)
    h2 = _cget(h, i, j, ny, nx)
    h3 = _cget(h, i, j + 1, ny, nx)
    wl = _dryfactor(h1, h0, h2, dry)  # left cell (j-1)
    wr = _dryfactor(h2, h1, h3, dry)  # right cell (j)

    # Left cell (j-1) reconstructed to its right face; right cell (j) to its left.
    eta_l = e1 + 0.5 * wl * _minmod(e1 - e0, e2 - e1)
    eta_r = e2 - 0.5 * wr * _minmod(e2 - e1, e3 - e2)
    zf_l = z1 + 0.5 * wl * _minmod(z1 - z0, z2 - z1)
    zf_r = z2 - 0.5 * wr * _minmod(z2 - z1, z3 - z2)
    u_l = u1 + 0.5 * wl * _minmod(u1 - u0, u2 - u1)
    u_r = u2 - 0.5 * wr * _minmod(u2 - u1, u3 - u2)
    v_l = v1 + 0.5 * wl * _minmod(v1 - v0, v2 - v1)
    v_r = v2 - 0.5 * wr * _minmod(v2 - v1, v3 - v2)

    h_l = wp.max(eta_l - zf_l, 0.0)
    h_r = wp.max(eta_r - zf_r, 0.0)
    z_star = wp.max(zf_l, zf_r)
    hl_star = wp.max(eta_l - z_star, 0.0)
    hr_star = wp.max(eta_r - z_star, 0.0)

    f = _hllc(hl_star, u_l, v_l, hr_star, u_r, v_r, g, dry)
    fx_h[i, j] = f[0]
    fx_mn_l[i, j] = f[1] + 0.5 * g * (h_l * h_l - hl_star * hl_star)
    fx_mn_r[i, j] = f[1] + 0.5 * g * (h_r * h_r - hr_star * hr_star)
    fx_mt[i, j] = f[2]


@wp.kernel
def _flux_y(
    h: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    uvel: wp.array2d(dtype=wp.float32),
    vvel: wp.array2d(dtype=wp.float32),
    fy_h: wp.array2d(dtype=wp.float32),
    fy_mn_l: wp.array2d(dtype=wp.float32),
    fy_mn_r: wp.array2d(dtype=wp.float32),
    fy_mt: wp.array2d(dtype=wp.float32),
    ny: wp.int32,
    nx: wp.int32,
    g: wp.float32,
    dry: wp.float32,
):
    """HLLC flux on y-face ``i`` (between cells ``(i-1, j)`` and ``(i, j)``).

    Launched over all faces ``(ny+1, nx)``. Normal velocity is ``v``, transverse
    is ``u`` (the roles of the two momenta swap versus the x-sweep).
    """
    i, j = wp.tid()
    e0 = _cget(eta, i - 2, j, ny, nx)
    e1 = _cget(eta, i - 1, j, ny, nx)
    e2 = _cget(eta, i, j, ny, nx)
    e3 = _cget(eta, i + 1, j, ny, nx)
    z0 = _cget(z, i - 2, j, ny, nx)
    z1 = _cget(z, i - 1, j, ny, nx)
    z2 = _cget(z, i, j, ny, nx)
    z3 = _cget(z, i + 1, j, ny, nx)
    u0 = _cget(uvel, i - 2, j, ny, nx)
    u1 = _cget(uvel, i - 1, j, ny, nx)
    u2 = _cget(uvel, i, j, ny, nx)
    u3 = _cget(uvel, i + 1, j, ny, nx)
    v0 = _cget(vvel, i - 2, j, ny, nx)
    v1 = _cget(vvel, i - 1, j, ny, nx)
    v2 = _cget(vvel, i, j, ny, nx)
    v3 = _cget(vvel, i + 1, j, ny, nx)

    # First-order at dry-adjacent cells (see _dryfactor); rows i-2..i+1.
    h0 = _cget(h, i - 2, j, ny, nx)
    h1 = _cget(h, i - 1, j, ny, nx)
    h2 = _cget(h, i, j, ny, nx)
    h3 = _cget(h, i + 1, j, ny, nx)
    wl = _dryfactor(h1, h0, h2, dry)  # left cell (i-1)
    wr = _dryfactor(h2, h1, h3, dry)  # right cell (i)

    eta_l = e1 + 0.5 * wl * _minmod(e1 - e0, e2 - e1)
    eta_r = e2 - 0.5 * wr * _minmod(e2 - e1, e3 - e2)
    zf_l = z1 + 0.5 * wl * _minmod(z1 - z0, z2 - z1)
    zf_r = z2 - 0.5 * wr * _minmod(z2 - z1, z3 - z2)
    # Normal velocity is v; transverse is u.
    vn_l = v1 + 0.5 * wl * _minmod(v1 - v0, v2 - v1)
    vn_r = v2 - 0.5 * wr * _minmod(v2 - v1, v3 - v2)
    ut_l = u1 + 0.5 * wl * _minmod(u1 - u0, u2 - u1)
    ut_r = u2 - 0.5 * wr * _minmod(u2 - u1, u3 - u2)

    h_l = wp.max(eta_l - zf_l, 0.0)
    h_r = wp.max(eta_r - zf_r, 0.0)
    z_star = wp.max(zf_l, zf_r)
    hl_star = wp.max(eta_l - z_star, 0.0)
    hr_star = wp.max(eta_r - z_star, 0.0)

    f = _hllc(hl_star, vn_l, ut_l, hr_star, vn_r, ut_r, g, dry)
    fy_h[i, j] = f[0]
    fy_mn_l[i, j] = f[1] + 0.5 * g * (h_l * h_l - hl_star * hl_star)
    fy_mn_r[i, j] = f[1] + 0.5 * g * (h_r * h_r - hr_star * hr_star)
    fy_mt[i, j] = f[2]


@wp.kernel
def _accumulate(
    h: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    z: wp.array2d(dtype=wp.float32),
    fx_h: wp.array2d(dtype=wp.float32),
    fx_mn_l: wp.array2d(dtype=wp.float32),
    fx_mn_r: wp.array2d(dtype=wp.float32),
    fx_mt: wp.array2d(dtype=wp.float32),
    fy_h: wp.array2d(dtype=wp.float32),
    fy_mn_l: wp.array2d(dtype=wp.float32),
    fy_mn_r: wp.array2d(dtype=wp.float32),
    fy_mt: wp.array2d(dtype=wp.float32),
    dh: wp.array2d(dtype=wp.float32),
    dhu: wp.array2d(dtype=wp.float32),
    dhv: wp.array2d(dtype=wp.float32),
    ny: wp.int32,
    nx: wp.int32,
    dx: wp.float32,
    g: wp.float32,
    dry: wp.float32,
):
    """Sum x/y flux divergences plus the centered bed-slope source -> ``dU/dt``."""
    i, j = wp.tid()
    inv_dx = 1.0 / dx

    # Mass: net flux through the four faces.
    dh[i, j] = -((fx_h[i, j + 1] - fx_h[i, j]) + (fy_h[i + 1, j] - fy_h[i, j])) * inv_dx

    # Wet/dry slope gates -- read from the SAME state ``h`` the flux kernels gate on
    # (via ``_cget(h, ...)``), so the first-order drop is byte-identical between the
    # source and the Audusse flux corrections. Gating on ``eta - z`` instead would
    # differ by ~ULP(z) in float32 (eta was formed as h+z), letting a marginal cell
    # be 2nd-order in the flux but 1st-order here -> a broken telescoping and an
    # O(0.1 m/s) residual at the shoreline. Same array => same gate, exactly.
    hg = _cget(h, i, j, ny, nx)
    wx = _dryfactor(hg, _cget(h, i, j - 1, ny, nx), _cget(h, i, j + 1, ny, nx), dry)
    wy = _dryfactor(hg, _cget(h, i - 1, j, ny, nx), _cget(h, i + 1, j, ny, nx), dry)

    # x-momentum (hu): normal flux through x-faces + transverse flux through y-faces
    # + the x centered bed-slope source.
    sz_x = wx * _minmod(
        _cget(z, i, j, ny, nx) - _cget(z, i, j - 1, ny, nx),
        _cget(z, i, j + 1, ny, nx) - _cget(z, i, j, ny, nx),
    )
    se_x = wx * _minmod(
        _cget(eta, i, j, ny, nx) - _cget(eta, i, j - 1, ny, nx),
        _cget(eta, i, j + 1, ny, nx) - _cget(eta, i, j, ny, nx),
    )
    hp_x = wp.max((eta[i, j] + 0.5 * se_x) - (z[i, j] + 0.5 * sz_x), 0.0)
    hm_x = wp.max((eta[i, j] - 0.5 * se_x) - (z[i, j] - 0.5 * sz_x), 0.0)
    src_x = -g * 0.5 * (hp_x + hm_x) * sz_x
    dhu[i, j] = (
        -((fx_mn_l[i, j + 1] - fx_mn_r[i, j]) + (fy_mt[i + 1, j] - fy_mt[i, j])) + src_x
    ) * inv_dx

    # y-momentum (hv): normal flux through y-faces + transverse flux through x-faces
    # + the y centered bed-slope source.
    sz_y = wy * _minmod(
        _cget(z, i, j, ny, nx) - _cget(z, i - 1, j, ny, nx),
        _cget(z, i + 1, j, ny, nx) - _cget(z, i, j, ny, nx),
    )
    se_y = wy * _minmod(
        _cget(eta, i, j, ny, nx) - _cget(eta, i - 1, j, ny, nx),
        _cget(eta, i + 1, j, ny, nx) - _cget(eta, i, j, ny, nx),
    )
    hp_y = wp.max((eta[i, j] + 0.5 * se_y) - (z[i, j] + 0.5 * sz_y), 0.0)
    hm_y = wp.max((eta[i, j] - 0.5 * se_y) - (z[i, j] - 0.5 * sz_y), 0.0)
    src_y = -g * 0.5 * (hp_y + hm_y) * sz_y
    dhv[i, j] = (
        -((fy_mn_l[i + 1, j] - fy_mn_r[i, j]) + (fx_mt[i, j + 1] - fx_mt[i, j])) + src_y
    ) * inv_dx


@wp.kernel
def _rk_stage1(
    h: wp.array2d(dtype=wp.float32),
    hu: wp.array2d(dtype=wp.float32),
    hv: wp.array2d(dtype=wp.float32),
    dh: wp.array2d(dtype=wp.float32),
    dhu: wp.array2d(dtype=wp.float32),
    dhv: wp.array2d(dtype=wp.float32),
    h1: wp.array2d(dtype=wp.float32),
    hu1: wp.array2d(dtype=wp.float32),
    hv1: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    dry: wp.float32,
):
    """Euler predictor ``U1 = U^n + dt L(U^n)`` (clamp h>=0, zero dry momentum)."""
    i, j = wp.tid()
    hh = wp.max(h[i, j] + dt * dh[i, j], 0.0)
    h1[i, j] = hh
    if hh > dry:
        hu1[i, j] = hu[i, j] + dt * dhu[i, j]
        hv1[i, j] = hv[i, j] + dt * dhv[i, j]
    else:
        hu1[i, j] = 0.0
        hv1[i, j] = 0.0


@wp.kernel
def _rk_stage2(
    h: wp.array2d(dtype=wp.float32),
    hu: wp.array2d(dtype=wp.float32),
    hv: wp.array2d(dtype=wp.float32),
    h1: wp.array2d(dtype=wp.float32),
    hu1: wp.array2d(dtype=wp.float32),
    hv1: wp.array2d(dtype=wp.float32),
    dh: wp.array2d(dtype=wp.float32),
    dhu: wp.array2d(dtype=wp.float32),
    dhv: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    dry: wp.float32,
):
    """Heun corrector ``U^{n+1} = 1/2 (U^n + U1 + dt L(U1))`` (in place on U^n)."""
    i, j = wp.tid()
    hh = wp.max(0.5 * (h[i, j] + h1[i, j] + dt * dh[i, j]), 0.0)
    h[i, j] = hh
    if hh > dry:
        hu[i, j] = 0.5 * (hu[i, j] + hu1[i, j] + dt * dhu[i, j])
        hv[i, j] = 0.5 * (hv[i, j] + hv1[i, j] + dt * dhv[i, j])
    else:
        hu[i, j] = 0.0
        hv[i, j] = 0.0


@wp.kernel
def _friction(
    h: wp.array2d(dtype=wp.float32),
    hu: wp.array2d(dtype=wp.float32),
    hv: wp.array2d(dtype=wp.float32),
    n: wp.array2d(dtype=wp.float32),
    g: wp.float32,
    dt: wp.float32,
    dry: wp.float32,
):
    """Semi-implicit Manning friction on the cell momentum (HANDOFF §8).

    Shares one definition of the denominator ``D`` with the LI scheme via
    :func:`solver.core.friction.manning_denominator`. The two are algebraically
    identical: LI passes discharge ``q = h*|vel|`` over ``h^(7/3)``; the cell
    momentum magnitude ``mom = sqrt(hu^2+hv^2)`` *is* that ``q``, and
    ``|q|/h^(7/3) = |vel|/h^(4/3)`` -- the momentum-form Manning slope.
    """
    i, j = wp.tid()
    hh = h[i, j]
    if hh <= dry:
        hu[i, j] = 0.0
        hv[i, j] = 0.0
        return
    mom = wp.sqrt(hu[i, j] * hu[i, j] + hv[i, j] * hv[i, j])  # = discharge |q| = h*|vel|
    denom = manning_denominator(mom, hh, n[i, j], g, dt)
    hu[i, j] = hu[i, j] / denom
    hv[i, j] = hv[i, j] / denom


@wp.kernel
def _add_uniform_rain(h: wp.array2d(dtype=wp.float32), rain: wp.float32, dt: wp.float32):
    i, j = wp.tid()
    h[i, j] = h[i, j] + rain * dt


@wp.kernel
def _reduce_wavespeed(
    h: wp.array2d(dtype=wp.float32),
    hu: wp.array2d(dtype=wp.float32),
    hv: wp.array2d(dtype=wp.float32),
    g: wp.float32,
    dry: wp.float32,
    out_max: wp.array(dtype=wp.float32),
):
    """Atomic-max of the cell wave speed ``max(|u|,|v|) + sqrt(g h)`` (deterministic)."""
    i, j = wp.tid()
    hh = h[i, j]
    c = wp.sqrt(g * wp.max(hh, 0.0))
    s = c
    if hh > dry:
        au = wp.abs(hu[i, j]) / hh
        av = wp.abs(hv[i, j]) / hh
        s = wp.max(au, av) + c
    wp.atomic_max(out_max, 0, s)


# --------------------------------------------------------------------------- #
# Scheme interface: compute_dt / step (mirrors solver.core.local_inertial).   #
# --------------------------------------------------------------------------- #
def compute_dt(state: State, alpha: float = 0.45, dt_max: float = 30.0) -> float:
    """State-derived CFL step ``dt = alpha * dx / max(|u|,|v| + sqrt(g h))`` (§8/§12).

    ``alpha`` is the HLLC CFL coefficient C (~0.4-0.5). Order-independent atomic-max
    reduction keeps ``dt`` reproducible. Returns ``dt_max`` on a dry/still domain.
    """
    arm_hllc(state)
    g = state.grid
    state.h_max.zero_()
    wp.launch(
        _reduce_wavespeed,
        dim=g.shape,
        inputs=[state.h, state.hu, state.hv, float(GRAVITY), float(H_DRY), state.h_max],
        device=state.device,
    )
    smax = float(state.h_max.numpy()[0])
    if smax <= 0.0:
        return dt_max
    return min(alpha * g.dx / smax, dt_max)


def _eval_L(state: State, h: wp.array, hu: wp.array, hv: wp.array) -> None:
    """Evaluate the spatial operator ``L(U) = dU/dt`` from (h, hu, hv) into scratch."""
    g = state.grid
    s: _HllcScratch = state.hllc
    dxf, gf, dryf = float(g.dx), float(GRAVITY), float(H_DRY)
    wp.launch(
        _primitives,
        dim=g.shape,
        inputs=[h, hu, hv, state.z, s.uvel, s.vvel, state.eta, dryf],
        device=state.device,
    )
    wp.launch(
        _flux_x,
        dim=g.qx_shape,
        inputs=[
            h,
            state.eta,
            state.z,
            s.uvel,
            s.vvel,
            s.fx_h,
            s.fx_mn_l,
            s.fx_mn_r,
            s.fx_mt,
            g.ny,
            g.nx,
            gf,
            dryf,
        ],
        device=state.device,
    )
    wp.launch(
        _flux_y,
        dim=g.qy_shape,
        inputs=[
            h,
            state.eta,
            state.z,
            s.uvel,
            s.vvel,
            s.fy_h,
            s.fy_mn_l,
            s.fy_mn_r,
            s.fy_mt,
            g.ny,
            g.nx,
            gf,
            dryf,
        ],
        device=state.device,
    )
    wp.launch(
        _accumulate,
        dim=g.shape,
        inputs=[
            h,
            state.eta,
            state.z,
            s.fx_h,
            s.fx_mn_l,
            s.fx_mn_r,
            s.fx_mt,
            s.fy_h,
            s.fy_mn_l,
            s.fy_mn_r,
            s.fy_mt,
            s.dh,
            s.dhu,
            s.dhv,
            g.ny,
            g.nx,
            dxf,
            gf,
            dryf,
        ],
        device=state.device,
    )


def step(
    state: State,
    dt: float,
    rain: float = 0.0,
    rain_scale: float = 1.0,
) -> None:
    """Advance one HLLC step of size ``dt`` (SSP-RK2 + friction + sources).

    Signature mirrors the local-inertial ``step`` so the run loop is scheme-agnostic
    (``rain`` uniform source; ``rain_scale`` gates the optional spatial rain field;
    ``state.infil`` the optional infiltration sink). Boundaries are transmissive
    (handled inside the flux kernels); closed/fixed-stage ghost BCs land later.
    """
    from solver.core.local_inertial import apply_infiltration, apply_rain_field

    arm_hllc(state)
    g = state.grid
    s: _HllcScratch = state.hllc
    dtf, gf, dryf = float(dt), float(GRAVITY), float(H_DRY)

    # SSP-RK2 (Heun): predictor then corrector, each a full L evaluation.
    _eval_L(state, state.h, state.hu, state.hv)
    wp.launch(
        _rk_stage1,
        dim=g.shape,
        inputs=[state.h, state.hu, state.hv, s.dh, s.dhu, s.dhv, s.h1, s.hu1, s.hv1, dtf, dryf],
        device=state.device,
    )
    _eval_L(state, s.h1, s.hu1, s.hv1)
    wp.launch(
        _rk_stage2,
        dim=g.shape,
        inputs=[state.h, state.hu, state.hv, s.h1, s.hu1, s.hv1, s.dh, s.dhu, s.dhv, dtf, dryf],
        device=state.device,
    )

    # Semi-implicit Manning friction (post-step operator split).
    wp.launch(
        _friction,
        dim=g.shape,
        inputs=[state.h, state.hu, state.hv, state.n, gf, dtf, dryf],
        device=state.device,
    )

    # Sources/sinks (mass only; momentum unchanged) -- reuse the LI kernels.
    if rain != 0.0:
        wp.launch(
            _add_uniform_rain, dim=g.shape, inputs=[state.h, float(rain), dtf], device=state.device
        )
    if state.rain is not None:
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
