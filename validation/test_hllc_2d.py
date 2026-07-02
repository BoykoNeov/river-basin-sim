"""2-D HLLC finite-volume scheme validation (M4 steps 5/6/8) on Warp's CPU backend.

Gates the Warp port of the well-balanced HLLC scheme:

* **Lake-at-rest** -- the well-balancedness keystone. A flat surface over a sloped
  and bumpy bed must stay flat with no spurious velocity. In float32 the residual
  is roundoff-limited (~1e-5 m/s here) rather than machine-eps -- confirmed
  roundoff, not imbalance: it scales with the *absolute* datum, only sqrt(n) with
  steps. A genuinely non-well-balanced scheme would drive O(g*slope*dt) ~ 1e-2+
  velocity, so the 1e-4 gate is discriminating. (See the M4 plan / datum note.)
* **Dam-break** vs the analytical Stoker solution -- HLLC captures the shock far
  better than local-inertial (nRMSE well below LI's 0.074).
* **Determinism** (HANDOFF §8/§12) -- two identical runs must match bit-for-bit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from solver.core import hllc
from solver.core.state import State
from validation.analytical import _stoker_star_depth, stoker, stoker_front_position

wp.init()
DEV = "cpu"


def _bumpy_bed(ny: int, nx: int, dx: float) -> np.ndarray:
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (0.5 * np.sin(xx * 0.2) + 0.002 * xx * dx + 0.3 * np.cos(yy * 0.3)).astype(np.float32)


def test_lake_at_rest_stays_flat():
    """KEYSTONE: flat surface over a sloped+bumpy bed stays flat (well-balanced)."""
    ny, nx, dx = 30, 40, 10.0
    z = _bumpy_bed(ny, nx, dx)
    eta0 = float(z.max()) + 2.0
    h0 = np.maximum(eta0 - z, 0.0).astype(np.float32)
    st = State.from_bed(z, dx=dx, depth=h0, manning=0.03, device=DEV)

    for _ in range(200):
        dt = hllc.compute_dt(st, alpha=0.45, dt_max=5.0)
        hllc.step(st, dt=dt)

    u, v = st.velocities_numpy()
    h = st.h.numpy()
    max_vel = max(float(np.abs(u).max()), float(np.abs(v).max()))
    surface_drift = float(np.abs((h + z) - eta0).max())
    print(f"\n[hllc lake-at-rest] max|u,v|={max_vel:.3e}  surface_drift={surface_drift:.3e}")
    assert np.isfinite(h).all()
    assert max_vel < 1e-4, f"spurious rest velocity {max_vel:.3e} -- not well-balanced"
    assert surface_drift < 1e-3, f"surface drifted {surface_drift:.3e} from flat"


def test_wet_bed_dam_break_beats_li():
    """1-row dam break: HLLC shock shape/front tighter than LI; mass conserved."""
    hL, hR = 1.0, 0.3
    nx, dx, t_end = 800, 0.5, 8.0
    x = (np.arange(nx) + 0.5 - nx / 2.0) * dx
    depth0 = np.where(x < 0.0, hL, hR).astype(np.float32).reshape(1, nx)
    bed = np.zeros((1, nx), dtype=np.float32)
    st = State.from_bed(bed, dx=dx, depth=depth0, manning=0.01, device=DEV)
    m0 = float(st.h.numpy().sum()) * dx

    t = 0.0
    while t < t_end - 1e-9:
        dt = hllc.compute_dt(st, alpha=0.45, dt_max=t_end)
        dt = min(dt, t_end - t)
        hllc.step(st, dt=dt)
        t += dt

    h = st.h.numpy()[0].astype(np.float64)
    assert np.isfinite(h).all() and h.min() >= 0.0
    # Waves stay interior -> mass conserved despite transmissive edges.
    assert abs(h.sum() * dx - m0) / m0 < 1e-5

    h_ref, _ = stoker(x, t_end, hL, hR)
    interior = np.abs(x) < 0.9 * (nx / 2.0 * dx)
    nrmse = math.sqrt(np.mean((h[interior] - h_ref[interior]) ** 2)) / (hL - hR)
    lvl = 0.5 * (hR + _stoker_star_depth(hL, hR))
    front_sim = x[h > lvl].max()
    front_err = abs(front_sim - stoker_front_position(t_end, hL, hR)) / stoker_front_position(
        t_end, hL, hR
    )
    print(f"\n[hllc 2D wet-bed] nRMSE={nrmse:.4f} (LI 0.074)  front_err={front_err:.4f}")
    assert nrmse < 0.03, f"HLLC nRMSE {nrmse:.4f} must beat LI's 0.074"
    assert front_err < 0.05


def test_hllc_is_bitwise_deterministic():
    """Determinism invariant (§12): two identical HLLC runs match bit-for-bit."""
    ny, nx, dx = 16, 20, 10.0
    z = _bumpy_bed(ny, nx, dx)
    yy, xx = np.mgrid[0:ny, 0:nx]
    depth0 = (0.5 + 0.3 * np.sin(xx * 0.5 + yy * 0.3)).astype(np.float32)  # non-uniform, moving

    def run() -> tuple[np.ndarray, np.ndarray]:
        st = State.from_bed(z, dx=dx, depth=depth0, manning=0.03, device=DEV)
        for _ in range(60):
            dt = hllc.compute_dt(st, alpha=0.45, dt_max=5.0)
            hllc.step(st, dt=dt, rain=1e-5)
        return st.h.numpy(), st.hu.numpy()

    h_a, hu_a = run()
    h_b, hu_b = run()
    assert np.array_equal(h_a, h_b)
    assert np.array_equal(hu_a, hu_b)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
