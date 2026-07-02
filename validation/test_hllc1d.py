"""1-D HLLC reference validation (M4 step 4): dam-break + well-balancedness.

These gate the pure-NumPy reference before the 2-D Warp port. HLLC captures the
shock far better than local-inertial (which drops advective acceleration), so the
dam-break wave-shape gates are *tight* here -- ~10x below LI's 0.074 nRMSE -- and
the 1-D lake-at-rest is machine-eps clean, confirming the Audusse hydrostatic
reconstruction is well-balanced with a static (flow-independent) bed slope.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from validation.analytical import _stoker_star_depth, ritter, stoker, stoker_front_position
from validation.hllc1d import G, hllc_flux, solve_1d


def _cell_x(nx: int, dx: float) -> np.ndarray:
    return (np.arange(nx) + 0.5 - nx / 2.0) * dx


def test_wet_bed_stoker_shock_and_mass():
    """Wet-bed dam break: mass exact, and shock shape/front far tighter than LI."""
    hL, hR = 1.0, 0.3
    nx, dx, t_end = 800, 0.5, 8.0
    x = _cell_x(nx, dx)
    h0 = np.where(x < 0.0, hL, hR).astype(float)
    z = np.zeros(nx)
    m0 = h0.sum() * dx

    h, u = solve_1d(h0, np.zeros(nx), z, dx, t_end, cfl=0.45, manning_n=0.01)

    # Mass conservation (closed-form: transmissive BCs, waves stay interior).
    assert np.isfinite(h).all() and h.min() >= 0.0
    assert abs(h.sum() * dx - m0) / m0 < 1e-12

    h_ref, _ = stoker(x, t_end, hL, hR)
    interior = np.abs(x) < 0.9 * (nx / 2.0 * dx)
    nrmse = math.sqrt(np.mean((h[interior] - h_ref[interior]) ** 2)) / (hL - hR)

    lvl = 0.5 * (hR + _stoker_star_depth(hL, hR))
    front_sim = x[h > lvl].max()
    front_ref = stoker_front_position(t_end, hL, hR)
    front_err = abs(front_sim - front_ref) / abs(front_ref)

    print(f"\n[hllc wet-bed] nRMSE={nrmse:.4f} (LI 0.074)  front_err={front_err:.4f}")
    assert nrmse < 0.03, f"HLLC nRMSE {nrmse:.4f} must beat LI's 0.074"
    assert front_err < 0.05, f"front error {front_err:.4f}"


def test_dry_bed_ritter_front_is_clean():
    """Dry-bed dam break: HLLC resolves the wetting front (LI's worst case) cleanly."""
    hL = 1.0
    nx, dx, t_end = 800, 0.5, 6.0
    x = _cell_x(nx, dx)
    h0 = np.where(x < 0.0, hL, 0.0).astype(float)

    h, u = solve_1d(h0, np.zeros(nx), np.zeros(nx), dx, t_end, cfl=0.45, manning_n=0.0)

    assert np.isfinite(h).all() and h.min() >= 0.0
    h_ref, _ = ritter(x, t_end, hL)
    interior = np.abs(x) < 0.9 * (nx / 2.0 * dx)
    nrmse = math.sqrt(np.mean((h[interior] - h_ref[interior]) ** 2)) / hL
    print(f"\n[hllc dry-bed] nRMSE={nrmse:.4f}")
    assert nrmse < 0.03


def test_lake_at_rest_is_machine_eps_on_a_bumpy_bed():
    """Well-balancedness keystone (1-D): flat surface over a sloped+bumpy bed stays
    flat to machine precision -- the discriminating test for hydrostatic recon."""
    nx, dx = 200, 1.0
    xc = (np.arange(nx) + 0.5) * dx
    z = 0.5 * np.sin(xc * 0.05) + 0.002 * xc + 0.3 * np.exp(-((xc - 100.0) ** 2) / 200.0)
    eta0 = z.max() + 1.0
    h0 = np.maximum(eta0 - z, 0.0)

    h, u = solve_1d(h0, np.zeros(nx), z, dx, 300.0, cfl=0.45, manning_n=0.0)

    assert np.abs(u).max() < 1e-10, f"lake-at-rest velocity {np.abs(u).max():.2e} not still"
    assert np.abs((h + z) - eta0).max() < 1e-10, "water surface drifted from flat"


def test_dry_over_dry_gives_zero_flux():
    """Two dry cells produce no flux (no NaN from the wave-speed estimate)."""
    Fh, Fhu, Fhv = hllc_flux(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert (Fh, Fhu, Fhv) == (0.0, 0.0, 0.0)


def test_flux_consistency_at_uniform_state():
    """F(U, U) equals the physical flux (Riemann-solver consistency)."""
    h, u, v = 2.0, 1.5, -0.7
    Fh, Fhu, Fhv = hllc_flux(h, u, v, h, u, v)
    assert Fh == pytest.approx(h * u)
    assert Fhu == pytest.approx(h * u * u + 0.5 * G * h * h)
    assert Fhv == pytest.approx(h * u * v)


def test_transverse_momentum_is_upwind_of_the_contact():
    """The contact wave carries hv from the upwind side (right-moving flow -> vL)."""
    # Strongly right-moving flow: contact speed > 0 -> transverse from the left.
    Fh, _, Fhv = hllc_flux(1.0, 3.0, 5.0, 1.0, 3.0, -5.0)
    assert Fhv == pytest.approx(Fh * 5.0)  # carries vL, not vR


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
