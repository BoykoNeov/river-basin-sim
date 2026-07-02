"""1-D well-balanced HLLC finite-volume reference solver (M4, HANDOFF §8).

A pure-NumPy reference for the M4 fidelity scheme, used to validate the numerics
(HLLC flux, MUSCL reconstruction, Audusse hydrostatic reconstruction, SSP-RK2,
semi-implicit friction) in 1-D *before* the 2-D Warp port -- cheap confidence, and
the algorithmic reference the Warp x-sweep mirrors (plan §3 step 4).

Conservative shallow water, state ``U = [h, hu, hv]`` (the transverse momentum
``hv`` is carried as a passive scalar so this same flux generalises directly to
the 2-D dimensional-split x-sweep). Flux ``F = [hu, hu^2 + g h^2/2, huv]``.

Well-balancedness (the property that makes M4 "well-balanced") comes from Audusse
et al. (2004) **hydrostatic reconstruction**: the *static bed* ``z`` is
reconstructed once from geometry (a flow-independent slope), the flow variables
(water surface ``eta = h + z`` and velocity ``u``) are reconstructed separately,
and the face depth is ``h_face = eta_face - z_face``. Reconstructing the *surface*
(not the depth) is what makes a lake at rest stay flat to machine precision.
Reconstructing ``z`` from geometry (not as ``eta - h``) keeps the bed slope in the
centered source term flow-independent -- correct in motion, not just at rest.

The MUSCL slope limiter is minmod (TVD, robust at wet/dry). HLLC reduces to HLL
for ``[h, hu]`` in 1-D; the contact wave advects the transverse momentum ``hv``
upwind of the contact speed (Toro). Wave-speed estimates use the two-rarefaction
guess with explicit dry-left / dry-right branches.
"""

from __future__ import annotations

import numpy as np

G = 9.81
H_DRY = 1.0e-3


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise minmod limiter: 0 unless a, b share a sign, else the smaller |.|."""
    return np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _phys_flux(h: np.ndarray, u: np.ndarray, v: np.ndarray, g: float) -> tuple:
    """Physical shallow-water x-flux components ``F = [hu, hu^2 + g h^2/2, huv]``."""
    hu = h * u
    return hu, hu * u + 0.5 * g * h * h, hu * v


def hllc_flux(
    hL: np.ndarray,
    uL: np.ndarray,
    vL: np.ndarray,
    hR: np.ndarray,
    uR: np.ndarray,
    vR: np.ndarray,
    g: float = G,
    dry: float = H_DRY,
) -> tuple:
    """HLLC numerical flux for shallow water (vectorised over faces).

    ``[h, hu]`` follow the HLL flux in the subsonic star region (HLLC == HLL for
    these in 1-D); the transverse momentum ``hv`` is advected by the mass flux,
    upwinded on the sign of the contact wave speed ``s*`` (Toro). Returns
    ``(Fh, Fhu, Fhv)``.
    """
    hL = np.asarray(hL, dtype=np.float64)
    hR = np.asarray(hR, dtype=np.float64)
    uL = np.asarray(uL, dtype=np.float64)
    uR = np.asarray(uR, dtype=np.float64)
    vL = np.asarray(vL, dtype=np.float64)
    vR = np.asarray(vR, dtype=np.float64)

    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))
    dryL = hL <= dry
    dryR = hR <= dry

    # Two-rarefaction wave-speed estimate for the wet/wet case (Toro).
    h_star = np.maximum((0.5 * (cL + cR) + 0.25 * (uL - uR)) ** 2 / g, 0.0)
    c_star = np.sqrt(g * h_star)
    u_star = 0.5 * (uL + uR) + cL - cR
    sL = np.minimum(uL - cL, u_star - c_star)
    sR = np.maximum(uR + cR, u_star + c_star)
    # Dry-bed branches: the wet side sends a rarefaction into the dry side.
    sL = np.where(dryR, uL - cL, sL)
    sR = np.where(dryR, uL + 2.0 * cL, sR)
    sL = np.where(dryL, uR - 2.0 * cR, sL)
    sR = np.where(dryL, uR + cR, sR)

    FhL, FhuL, _ = _phys_flux(hL, uL, vL, g)
    FhR, FhuR, _ = _phys_flux(hR, uR, vR, g)
    UhL, UhuL = hL, hL * uL
    UhR, UhuR = hR, hR * uR

    denom = np.where(sR - sL != 0.0, sR - sL, 1.0)
    Fh_hll = (sR * FhL - sL * FhR + sL * sR * (UhR - UhL)) / denom
    Fhu_hll = (sR * FhuL - sL * FhuR + sL * sR * (UhuR - UhuL)) / denom

    # Contact (middle) wave speed; falls back to the average when degenerate.
    den_s = hR * (uR - sR) - hL * (uL - sL)
    s_star = np.where(
        np.abs(den_s) > 1e-12,
        (sL * hR * (uR - sR) - sR * hL * (uL - sL)) / np.where(den_s != 0.0, den_s, 1.0),
        0.5 * (uL + uR),
    )

    Fh = np.select([sL >= 0.0, sR <= 0.0], [FhL, FhR], default=Fh_hll)
    Fhu = np.select([sL >= 0.0, sR <= 0.0], [FhuL, FhuR], default=Fhu_hll)
    # Transverse momentum rides the mass flux, upwind of the contact. In the
    # supersonic cases s* shares the sign of the flow, so this equals the physical
    # flux there too -- one branch handles all three regions.
    Fhv = Fh * np.where(s_star >= 0.0, vL, vR)

    both_dry = dryL & dryR
    Fh = np.where(both_dry, 0.0, Fh)
    Fhu = np.where(both_dry, 0.0, Fhu)
    Fhv = np.where(both_dry, 0.0, Fhv)
    return Fh, Fhu, Fhv


def _spatial_operator(
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    z: np.ndarray,
    dx: float,
    g: float,
    dry: float,
) -> tuple:
    """Well-balanced MUSCL/HLLC spatial operator ``L(U) = dU/dt`` (transmissive BC).

    Static bed reconstructed once from geometry; flow reconstructed on ``eta``/``u``
    (advisor's fix -- keeps the centered bed-slope source flow-independent). Ghost
    cells (edge-replicated) give the transmissive boundary and the MUSCL stencil.
    """
    u = np.where(h > dry, hu / np.maximum(h, dry), 0.0)
    v = np.where(h > dry, hv / np.maximum(h, dry), 0.0)
    eta = h + z

    # Pad with two ghost cells each side (transmissive = edge replicate).
    up = np.pad(u, 2, mode="edge")
    vp = np.pad(v, 2, mode="edge")
    etap = np.pad(eta, 2, mode="edge")
    zp = np.pad(z, 2, mode="edge")

    def slope(q: np.ndarray) -> np.ndarray:
        return _minmod(q[1:-1] - q[:-2], q[2:] - q[1:-1])  # real cells + 1 ghost each side

    s_eta = slope(etap)
    s_u = slope(up)
    s_z = slope(zp)  # static bed slope -- flow-independent (the fix)

    etac = etap[1:-1]
    uc = up[1:-1]
    vc = vp[1:-1]
    zc = zp[1:-1]

    # Face reconstructions (+ = right face of a cell, - = left face).
    z_plus, z_minus = zc + 0.5 * s_z, zc - 0.5 * s_z
    eta_plus, eta_minus = etac + 0.5 * s_eta, etac - 0.5 * s_eta
    u_plus, u_minus = uc + 0.5 * s_u, uc - 0.5 * s_u
    h_plus = np.maximum(eta_plus - z_plus, 0.0)
    h_minus = np.maximum(eta_minus - z_minus, 0.0)

    # Interface k is between cell (k-1)'s right face and cell k's left face.
    hL, uL, zL, etaL = h_plus[:-1], u_plus[:-1], z_plus[:-1], eta_plus[:-1]
    hR, uR, zR, etaR = h_minus[1:], u_minus[1:], z_minus[1:], eta_minus[1:]
    vL, vR = vc[:-1], vc[1:]

    # Audusse hydrostatic reconstruction: raise the interface bed to the higher of
    # the two, so the pressure flux and bed source cancel exactly at rest.
    z_star = np.maximum(zL, zR)
    hL_star = np.maximum(etaL - z_star, 0.0)
    hR_star = np.maximum(etaR - z_star, 0.0)

    Fh, Fhu, Fhv = hllc_flux(hL_star, uL, vL, hR_star, uR, vR, g, dry)
    # Momentum pressure corrections restore the pressure at the true edge depth.
    FL_hu = Fhu + 0.5 * g * (hL**2 - hL_star**2)
    FR_hu = Fhu + 0.5 * g * (hR**2 - hR_star**2)

    inv_dx = 1.0 / dx
    dh = -(Fh[1:] - Fh[:-1]) * inv_dx
    dhu = -(FL_hu[1:] - FR_hu[:-1]) * inv_dx
    dhv = -(Fhv[1:] - Fhv[:-1]) * inv_dx

    # Centered bed-slope source (Audusse 2nd-order): uses the *static* bed slope, so
    # it cancels the pressure-flux imbalance both at rest and in motion.
    h_plus_r, h_minus_r, s_z_r = h_plus[1:-1], h_minus[1:-1], s_z[1:-1]
    dhu += (-g * 0.5 * (h_plus_r + h_minus_r) * s_z_r) * inv_dx
    return dh, dhu, dhv


def compute_dt(h: np.ndarray, hu: np.ndarray, dx: float, cfl: float, g: float, dry: float) -> float:
    """State-derived CFL step ``dt = cfl * dx / max(|u| + sqrt(g h))``."""
    u = np.where(h > dry, hu / np.maximum(h, dry), 0.0)
    smax = float(np.max(np.abs(u) + np.sqrt(g * np.maximum(h, 0.0))))
    return dx / 1.0 if smax <= 0.0 else cfl * dx / smax


def solve_1d(
    h0: np.ndarray,
    u0: np.ndarray,
    z: np.ndarray,
    dx: float,
    t_end: float,
    *,
    cfl: float = 0.45,
    manning_n: float = 0.0,
    g: float = G,
    dry: float = H_DRY,
    v0: np.ndarray | None = None,
) -> tuple:
    """Integrate the 1-D well-balanced HLLC solver to ``t_end``; return ``(h, u)``.

    SSP-RK2 (Heun) in time with a non-negativity clamp and dry-cell momentum reset;
    semi-implicit Manning friction applied as a per-step operator split.
    """
    h = np.asarray(h0, dtype=np.float64).copy()
    hu = (np.asarray(h0, dtype=np.float64) * np.asarray(u0, dtype=np.float64)).copy()
    hv = (
        np.zeros_like(h)
        if v0 is None
        else (np.asarray(h0, np.float64) * np.asarray(v0, np.float64))
    )
    z = np.asarray(z, dtype=np.float64)

    t = 0.0
    while t < t_end - 1e-12:
        dt = min(compute_dt(h, hu, dx, cfl, g, dry), t_end - t)

        dh1, dhu1, dhv1 = _spatial_operator(h, hu, hv, z, dx, g, dry)
        h1 = np.maximum(h + dt * dh1, 0.0)
        hu1, hv1 = hu + dt * dhu1, hv + dt * dhv1
        dh2, dhu2, dhv2 = _spatial_operator(h1, hu1, hv1, z, dx, g, dry)
        h = np.maximum(0.5 * (h + h1 + dt * dh2), 0.0)
        hu = 0.5 * (hu + hu1 + dt * dhu2)
        hv = 0.5 * (hv + hv1 + dt * dhv2)

        dry_mask = h <= dry
        hu[dry_mask] = 0.0
        hv[dry_mask] = 0.0

        if manning_n > 0.0:
            speed = np.sqrt(hu * hu + hv * hv) / np.maximum(h, dry)
            d_fric = 1.0 + g * manning_n * manning_n * dt * speed / np.maximum(h, dry) ** (
                4.0 / 3.0
            )
            hu = np.where(dry_mask, 0.0, hu / d_fric)
            hv = np.where(dry_mask, 0.0, hv / d_fric)

        t += dt

    u = np.where(h > dry, hu / np.maximum(h, dry), 0.0)
    return h, u
