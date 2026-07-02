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

**Boundaries are per-edge ghost cells (M4 step 9), read from ``state.boundaries``.**
The interior flux kernels compute *every* face -- boundary faces included -- with
edge-clamped (zero-gradient) neighbours, which is exactly the **transmissive /
free-outflow** ghost. Two per-edge corrections then run before the flux
divergence is accumulated:

  * **closed** edge -> a *reflective wall*: the boundary face flux is recomputed
    from an explicit ghost with the normal velocity negated (``u_ghost = -u_edge``,
    depth/surface/transverse copied). By antisymmetry the mass flux is exactly 0
    (a wall) and the transverse flux with it; the normal-momentum flux is the wall
    pressure. At rest ``u = 0`` so the reflected flux is *identical* to the
    transmissive one -- lake-at-rest is preserved by construction, the closed wall
    only changes anything in motion.
  * **open** edge -> transmissive (the clamped flux already computed) **plus mass
    banking**: the depth crossing the boundary face over the SSP-RK2 step is
    ``0.5*dt*(F_stage1 + F_stage2)/dx`` (the Heun weights; ``loss_cum`` holds a
    per-cell depth), banked into ``state.loss_cum`` so the float64 mass ledger stays
    balanced when water actually leaves. This is exact *provided the depth clamp
    never fires on a
    boundary cell*; it does not for a steady flow (gated in validation) but will in
    a drain-to-empty run (the EA cases, M4 step 10) -- a known limitation carried
    forward.

``fixed_stage`` (a prescribed-surface Dirichlet ghost) is additive to this
structure but needs a numeric per-edge config extension; deferred (plan §6,
non-gating). Determinism (HANDOFF §8/§12): every kernel writes each output from one
thread reading only inputs; the timestep is a state-derived atomic-max reduction,
order-independent like the LI scheme's.
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


# --------------------------------------------------------------------------- #
# Boundary corrections (M4 step 9). Run after the interior flux kernels, which  #
# have already filled every face (boundary faces via clamped = transmissive     #
# ghosts). Closed edges overwrite the boundary face with a reflective-wall flux; #
# open edges are left transmissive and banked separately (see _bank_* / step).   #
#                                                                               #
# Each wall kernel builds the ghost explicitly: normal velocity negated, depth / #
# transverse velocity copied. z_ghost = z_edge, so the Audusse star depth equals #
# the edge depth and the hydrostatic bed correction vanishes -- the boundary     #
# face's normal-momentum flux is just the HLLC pressure. The edge cell is already #
# first-order there (its outward neighbour is the clamped ghost), so the wall is  #
# order-consistent with the interior reconstruction and the balance holds. Only   #
# the *inward-facing* momentum slot is set (fx_mn_r at west / fx_mn_l at east):   #
# the other is read only by the nonexistent cell across the wall.               #
# --------------------------------------------------------------------------- #
@wp.kernel
def _wall_x_west(
    h: wp.array2d(dtype=wp.float32),
    uvel: wp.array2d(dtype=wp.float32),
    vvel: wp.array2d(dtype=wp.float32),
    fx_h: wp.array2d(dtype=wp.float32),
    fx_mn_r: wp.array2d(dtype=wp.float32),
    fx_mt: wp.array2d(dtype=wp.float32),
    g: wp.float32,
    dry: wp.float32,
):
    """West wall (x-face col 0): ghost = reflected cell (i, 0), interior on the right."""
    i = wp.tid()
    h0 = h[i, 0]
    f = _hllc(h0, -uvel[i, 0], vvel[i, 0], h0, uvel[i, 0], vvel[i, 0], g, dry)
    fx_h[i, 0] = f[0]  # = 0 by antisymmetry (no through-flux)
    fx_mn_r[i, 0] = f[1]  # wall pressure (bed correction vanishes: z_star = z0)
    fx_mt[i, 0] = f[2]  # = f[0]*vt = 0


@wp.kernel
def _wall_x_east(
    h: wp.array2d(dtype=wp.float32),
    uvel: wp.array2d(dtype=wp.float32),
    vvel: wp.array2d(dtype=wp.float32),
    fx_h: wp.array2d(dtype=wp.float32),
    fx_mn_l: wp.array2d(dtype=wp.float32),
    fx_mt: wp.array2d(dtype=wp.float32),
    nx: wp.int32,
    g: wp.float32,
    dry: wp.float32,
):
    """East wall (x-face col nx): interior cell (i, nx-1) on the left, reflected ghost right."""
    i = wp.tid()
    h0 = h[i, nx - 1]
    f = _hllc(h0, uvel[i, nx - 1], vvel[i, nx - 1], h0, -uvel[i, nx - 1], vvel[i, nx - 1], g, dry)
    fx_h[i, nx] = f[0]
    fx_mn_l[i, nx] = f[1]
    fx_mt[i, nx] = f[2]


@wp.kernel
def _wall_y_north(
    h: wp.array2d(dtype=wp.float32),
    uvel: wp.array2d(dtype=wp.float32),
    vvel: wp.array2d(dtype=wp.float32),
    fy_h: wp.array2d(dtype=wp.float32),
    fy_mn_r: wp.array2d(dtype=wp.float32),
    fy_mt: wp.array2d(dtype=wp.float32),
    g: wp.float32,
    dry: wp.float32,
):
    """North wall (y-face row 0): normal velocity is v, transverse is u (see _flux_y)."""
    j = wp.tid()
    h0 = h[0, j]
    f = _hllc(h0, -vvel[0, j], uvel[0, j], h0, vvel[0, j], uvel[0, j], g, dry)
    fy_h[0, j] = f[0]
    fy_mn_r[0, j] = f[1]
    fy_mt[0, j] = f[2]


@wp.kernel
def _wall_y_south(
    h: wp.array2d(dtype=wp.float32),
    uvel: wp.array2d(dtype=wp.float32),
    vvel: wp.array2d(dtype=wp.float32),
    fy_h: wp.array2d(dtype=wp.float32),
    fy_mn_l: wp.array2d(dtype=wp.float32),
    fy_mt: wp.array2d(dtype=wp.float32),
    ny: wp.int32,
    g: wp.float32,
    dry: wp.float32,
):
    """South wall (y-face row ny): interior cell (ny-1, j) on the left, reflected ghost below."""
    j = wp.tid()
    h0 = h[ny - 1, j]
    f = _hllc(h0, vvel[ny - 1, j], uvel[ny - 1, j], h0, -vvel[ny - 1, j], uvel[ny - 1, j], g, dry)
    fy_h[ny, j] = f[0]
    fy_mn_l[ny, j] = f[1]
    fy_mt[ny, j] = f[2]


# Open-edge mass banking (M4 step 9). The depth added to an edge cell from its
# boundary face over one SSP-RK2 step is 0.5*dt*(F_stage1 + F_stage2)/dx (Heun
# weights; loss_cum holds a per-cell depth); banking the negative of that records
# the outflow (signed, so a transmissive edge that draws water *in* banks a
# negative loss the ledger reads as inflow). ``wt`` carries 0.5*dt/dx for one
# stage. The +/- and which face column/row is read differ per edge with the flux
# sign convention (§grid).
@wp.kernel
def _bank_x_west(
    fx_h: wp.array2d(dtype=wp.float32),
    loss_cum: wp.array2d(dtype=wp.float64),
    wt: wp.float64,
):
    i = wp.tid()  # +fx_h[i,0] adds to cell 0 -> banked loss is its negative
    loss_cum[i, 0] = loss_cum[i, 0] - wt * wp.float64(fx_h[i, 0])


@wp.kernel
def _bank_x_east(
    fx_h: wp.array2d(dtype=wp.float32),
    loss_cum: wp.array2d(dtype=wp.float64),
    wt: wp.float64,
    nx: wp.int32,
):
    i = wp.tid()  # -fx_h[i,nx] adds to cell nx-1 -> banked loss is +fx_h[i,nx]
    loss_cum[i, nx - 1] = loss_cum[i, nx - 1] + wt * wp.float64(fx_h[i, nx])


@wp.kernel
def _bank_y_north(
    fy_h: wp.array2d(dtype=wp.float32),
    loss_cum: wp.array2d(dtype=wp.float64),
    wt: wp.float64,
):
    j = wp.tid()
    loss_cum[0, j] = loss_cum[0, j] - wt * wp.float64(fy_h[0, j])


@wp.kernel
def _bank_y_south(
    fy_h: wp.array2d(dtype=wp.float32),
    loss_cum: wp.array2d(dtype=wp.float64),
    wt: wp.float64,
    ny: wp.int32,
):
    j = wp.tid()
    loss_cum[ny - 1, j] = loss_cum[ny - 1, j] + wt * wp.float64(fy_h[ny, j])


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


def _apply_walls(state: State, h: wp.array) -> None:
    """Overwrite closed-edge boundary face fluxes with reflective-wall fluxes.

    Runs inside :func:`_eval_L` after the interior flux kernels (which filled the
    boundary faces transmissively) and before the flux divergence. ``h`` and the
    scratch velocities ``s.uvel/s.vvel`` are the current RK-stage state. Open edges
    are skipped -- they keep the transmissive flux and are banked in :func:`step`.
    """
    g = state.grid
    s: _HllcScratch = state.hllc
    bc = state.boundaries
    gf, dryf = float(GRAVITY), float(H_DRY)
    if bc["west"] == "closed":
        wp.launch(
            _wall_x_west,
            dim=g.ny,
            inputs=[h, s.uvel, s.vvel, s.fx_h, s.fx_mn_r, s.fx_mt, gf, dryf],
            device=state.device,
        )
    if bc["east"] == "closed":
        wp.launch(
            _wall_x_east,
            dim=g.ny,
            inputs=[h, s.uvel, s.vvel, s.fx_h, s.fx_mn_l, s.fx_mt, g.nx, gf, dryf],
            device=state.device,
        )
    if bc["north"] == "closed":
        wp.launch(
            _wall_y_north,
            dim=g.nx,
            inputs=[h, s.uvel, s.vvel, s.fy_h, s.fy_mn_r, s.fy_mt, gf, dryf],
            device=state.device,
        )
    if bc["south"] == "closed":
        wp.launch(
            _wall_y_south,
            dim=g.nx,
            inputs=[h, s.uvel, s.vvel, s.fy_h, s.fy_mn_l, s.fy_mt, g.ny, gf, dryf],
            device=state.device,
        )


def _bank_open_outflow(state: State, wt: float) -> None:
    """Bank one SSP-RK2 stage of open-edge boundary flux into ``loss_cum``.

    ``wt = 0.5*dt/dx`` (a Heun stage weight; ``loss_cum`` holds a per-cell depth);
    called once after each of the two per-step ``_eval_L`` evaluations, reading that
    stage's boundary mass flux. A no-op unless the scenario has an open edge (which
    armed ``loss_cum``).
    """
    if state.loss_cum is None:
        return
    g = state.grid
    s: _HllcScratch = state.hllc
    bc = state.boundaries
    wt64 = float(wt)
    if bc["west"] == "open":
        wp.launch(
            _bank_x_west, dim=g.ny, inputs=[s.fx_h, state.loss_cum, wt64], device=state.device
        )
    if bc["east"] == "open":
        wp.launch(
            _bank_x_east, dim=g.ny, inputs=[s.fx_h, state.loss_cum, wt64, g.nx], device=state.device
        )
    if bc["north"] == "open":
        wp.launch(
            _bank_y_north, dim=g.nx, inputs=[s.fy_h, state.loss_cum, wt64], device=state.device
        )
    if bc["south"] == "open":
        wp.launch(
            _bank_y_south,
            dim=g.nx,
            inputs=[s.fy_h, state.loss_cum, wt64, g.ny],
            device=state.device,
        )


def _eval_L(state: State, h: wp.array, hu: wp.array, hv: wp.array) -> None:
    """Evaluate the spatial operator ``L(U) = dU/dt`` from (h, hu, hv) into scratch.

    Closed-edge walls are applied to the boundary face fluxes before the divergence
    (:func:`_apply_walls`); open edges keep the transmissive flux and are banked by
    the caller after this returns (:func:`_bank_open_outflow`).
    """
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
    # Closed-edge reflective walls overwrite their boundary face fluxes; open edges
    # keep the transmissive flux computed above (banked by the caller).
    _apply_walls(state, h)
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
    ``state.infil`` the optional infiltration sink). Per-edge ghost-cell BCs come
    from ``state.boundaries`` (§9): closed edges are reflective walls (applied in
    ``_eval_L``), open edges are transmissive and mass-banked here into
    ``state.loss_cum``; ``fixed_stage`` is deferred (plan §6).
    """
    from solver.core.local_inertial import apply_infiltration, apply_rain_field

    arm_hllc(state)
    g = state.grid
    s: _HllcScratch = state.hllc
    dtf, dxf, gf, dryf = float(dt), float(g.dx), float(GRAVITY), float(H_DRY)

    # SSP-RK2 (Heun): predictor then corrector, each a full L evaluation. Open-edge
    # outflow is banked after each L eval with the matching Heun weight (0.5*dt),
    # so loss_cum tracks exactly the depth the flux divergence removed (§9).
    # loss_cum holds a per-cell *depth* (m); the boundary face removes depth
    # 0.5*dt*(fx_h*inv_dx) over the two Heun stages, so the stage weight is
    # 0.5*dt/dx (NOT *dx -- that would over-bank by dx^2).
    wt = 0.5 * dtf / dxf
    _eval_L(state, state.h, state.hu, state.hv)
    _bank_open_outflow(state, wt)
    wp.launch(
        _rk_stage1,
        dim=g.shape,
        inputs=[state.h, state.hu, state.hv, s.dh, s.dhu, s.dhv, s.h1, s.hu1, s.hv1, dtf, dryf],
        device=state.device,
    )
    _eval_L(state, s.h1, s.hu1, s.hv1)
    _bank_open_outflow(state, wt)
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
