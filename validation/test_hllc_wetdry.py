"""HLLC wet/dry + friction validation (M4 step 7) on Warp's CPU backend.

Step 6 landed the well-balanced *submerged* keystone (a fully-wet lake at rest);
step 8 the shock capturing (wet-bed dam break). This module gates the pieces in
between -- the ones that turn a Riemann solver into a robust flood scheme:

* **Shoreline lake-at-rest on a bumpy bed** -- the discriminating well-balancedness
  gate. The step-6 lake-at-rest is *fully wet*, so it never exercises a wet/dry
  front; a scheme can be exactly balanced there yet blow up the instant a bed bump
  pierces the surface (this scheme did: a smooth bowl spun up to 20 m/s before the
  step-7 dry-front fix). A pool with several bumps poking through as dry islands --
  many internal shorelines -- must stay at rest to ~1e-6, drift no surface, leak no
  water onto the islands, and lose no mass. This is what proves the first-order
  drop at wet/dry faces (``hllc._dryfactor``) keeps the Audusse balance intact.
* **Puddle in a depression at rest** -- a single smooth shoreline (a parabolic
  bowl), the minimal dry-front case (HANDOFF §12, the classic NaN source).
* **Dry-bed Ritter dam break** -- a rarefaction running onto a dry bed: depth stays
  non-negative, the wetting front tracks ``2 c_l t`` (loosely -- the dry threshold
  truncates the vanishing tip), and the fan matches the analytical profile.
* **Friction damps momentum** -- a moving slab on a flat bed decays monotonically
  toward rest without sign reversal (semi-implicit stability); a frictionless
  control does not decay.

**Deferred to step 9:** the *Manning normal-depth* check (the fidelity analogue of
the M3 local-inertial channel test). It needs a spatially-varying steady flow, which
develops a boundary-driven drawdown under HLLC's transmissive-on-``eta`` edges (a
uniform-depth flow on a slope has non-uniform ``eta``, so zero-gradient at the edge
is wrong there -- extrapolating the ghost bed does not fix it). The M3 channel test
sidesteps this with inflow + open ghost-cell BCs; the HLLC parallel lands with those
in step 9.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from solver.core import hllc
from solver.core.grid import GRAVITY, H_DRY
from solver.core.state import State
from validation.analytical import ritter

wp.init()
DEV = "cpu"


def test_shoreline_lake_at_rest_on_bumpy_bed():
    """A pool with dry islands (bumpy bed, many shorelines) stays at rest to ~1e-6.

    The discriminating gate the fully-wet step-6 lake-at-rest cannot make: the
    well-balanced cancellation must hold *cell-by-cell across every wet/dry front*,
    not just in the interior. Before the step-7 first-order-at-dry-front fix
    (``hllc._dryfactor``) a single smooth bowl already reached ~20 m/s here; a bumpy
    bed with several piercing bumps is stricter still. Balanced <=> the spurious
    velocity stays at the float32 at-rest floor.
    """
    ny, nx, dx = 50, 50, 5.0
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    # A gentle basin (dry, high at the edges) with three Gaussian bumps; some pierce
    # the fill level as dry islands, the valleys between them stay wet.
    z = 0.012 * ((xx - 24.5) ** 2 + (yy - 24.5) ** 2)
    z += 2.6 * np.exp(-(((xx - 16) ** 2 + (yy - 27) ** 2) / 22.0))
    z += 3.2 * np.exp(-(((xx - 33) ** 2 + (yy - 21) ** 2) / 26.0))
    z += 2.2 * np.exp(-(((xx - 27) ** 2 + (yy - 34) ** 2) / 18.0))
    z = z.astype(np.float32)
    fill = 3.0
    h0 = np.maximum(fill - z, 0.0).astype(np.float32)
    wet0 = h0 >= H_DRY
    dry0 = ~wet0

    # The pool must not touch the domain edge (keep it off the transmissive BCs).
    assert not (wet0[0, :].any() or wet0[-1, :].any() or wet0[:, 0].any() or wet0[:, -1].any())
    # There must be genuine *internal* shorelines: dry island cells with wet
    # neighbours, not just a single outer waterline.
    island_shore = dry0[1:-1, 1:-1] & (
        wet0[:-2, 1:-1] | wet0[2:, 1:-1] | wet0[1:-1, :-2] | wet0[1:-1, 2:]
    )
    assert island_shore.sum() > 50, "bed is not bumpy enough to make internal shorelines"

    st = State.from_bed(z, dx=dx, depth=h0, manning=0.03, device=DEV)
    v0 = float(st.h.numpy().sum())

    peak_vel = 0.0
    for _ in range(100):
        dt = hllc.compute_dt(st, alpha=0.45, dt_max=5.0)
        hllc.step(st, dt=dt)
        u, v = st.velocities_numpy()
        peak_vel = max(peak_vel, float(np.abs(u).max()), float(np.abs(v).max()))

    h = st.h.numpy()
    surface_drift = float(np.abs((h + z)[wet0] - fill).max())
    dry_leak = float(h[dry0].max())
    print(
        f"\n[hllc shoreline-lake] islands={int(island_shore.sum())} peak|u,v|={peak_vel:.3e}"
        f"  surface_drift={surface_drift:.3e}  dry_leak={dry_leak:.3e}"
        f"  dV={abs(float(h.sum()) - v0):.3e}"
    )
    assert np.isfinite(h).all() and h.min() >= 0.0
    # Tight, discriminating. The residual is a float32 round-off random walk: peak_vel
    # grows like sqrt(N_steps) (~1e-6 per sqrt-step, checked at N=50..600), the healthy
    # signature -- *linear* growth would signal a systematic imbalance. The pre-fix
    # scheme blew past this by ~6 orders (a smooth bowl alone reached ~20 m/s).
    assert peak_vel < 5e-5, f"lake-at-rest broke at a shoreline: {peak_vel:.3e} m/s"
    assert surface_drift < 1e-4, f"water surface drifted {surface_drift:.3e} m"
    assert dry_leak < H_DRY, f"water climbed {dry_leak:.3e} m onto a dry island"
    assert abs(float(h.sum()) - v0) < 1e-6 * v0, "lake lost mass"


def test_puddle_in_depression_stays_put():
    """A bowl of water with a dry rim stays at rest (dry-front well-balancedness).

    Discriminating over the submerged lake-at-rest: the wet/dry interface is where
    hydrostatic reconstruction must suppress spurious fronts. Water must not climb
    above the fill level into initially-dry cells, and none may be lost.
    """
    ny, nx, dx = 40, 40, 5.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    # Parabolic bowl, deepest (z=0) at the centre, rising to the rim.
    z = (0.04 * ((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2)).astype(np.float32)
    fill = 4.0  # water surface well below the domain-edge bed height
    h0 = np.maximum(fill - z, 0.0).astype(np.float32)
    wet0 = h0 >= H_DRY
    dry0 = ~wet0
    assert dry0.any() and wet0.any(), "test needs both wet and dry cells"
    # The puddle must not touch the domain edge (keep it clear of transmissive BCs).
    assert not (wet0[0, :].any() or wet0[-1, :].any() or wet0[:, 0].any() or wet0[:, -1].any())

    st = State.from_bed(z, dx=dx, depth=h0, manning=0.03, device=DEV)
    v0 = float(st.h.numpy().sum())

    for _ in range(300):
        dt = hllc.compute_dt(st, alpha=0.45, dt_max=5.0)
        hllc.step(st, dt=dt)

    h = st.h.numpy()
    u, v = st.velocities_numpy()
    max_vel = max(float(np.abs(u).max()), float(np.abs(v).max()))
    surface_drift = float(np.abs((h + z)[wet0] - fill).max())
    dry_leak = float(h[dry0].max())
    print(
        f"\n[hllc puddle] max|u,v|={max_vel:.3e}  surface_drift={surface_drift:.3e}"
        f"  dry_leak={dry_leak:.3e}  dV={abs(float(h.sum()) - v0):.3e}"
    )
    assert np.isfinite(h).all() and h.min() >= 0.0
    assert max_vel < 1e-4, f"puddle developed spurious velocity {max_vel:.3e}"
    assert surface_drift < 1e-3, f"puddle surface drifted {surface_drift:.3e}"
    assert dry_leak < H_DRY, f"water climbed {dry_leak:.3e} m into a dry cell above the rim"
    assert abs(float(h.sum()) - v0) < 1e-4 * v0, "puddle lost mass"


def test_dry_bed_ritter_dam_break():
    """Rarefaction onto a dry bed: non-negative, front near ``2 c_l t``, fan matches.

    The dry threshold ``H_DRY`` truncates the vanishing tip, so the simulated front
    lags the analytical one -- the assertion is a band (not ahead, not far behind),
    and the profile match is scored over the clearly-wetted fan only.
    """
    hL = 1.0
    nx, dx, t_end = 800, 0.5, 6.0
    x = (np.arange(nx) + 0.5 - nx / 2.0) * dx
    depth0 = np.where(x < 0.0, hL, 0.0).astype(np.float32).reshape(1, nx)
    bed = np.zeros((1, nx), dtype=np.float32)
    st = State.from_bed(bed, dx=dx, depth=depth0, manning=0.0, device=DEV)
    m0 = float(st.h.numpy().sum()) * dx

    t = 0.0
    while t < t_end - 1e-9:
        dt = hllc.compute_dt(st, alpha=0.45, dt_max=t_end)
        dt = min(dt, t_end - t)
        hllc.step(st, dt=dt)
        t += dt

    h = st.h.numpy()[0].astype(np.float64)
    assert np.isfinite(h).all() and h.min() >= 0.0

    c_l = math.sqrt(GRAVITY * hL)
    front_ana = 2.0 * c_l * t_end
    assert front_ana < 0.9 * (nx / 2.0 * dx), "front left the interior; shorten t_end"
    # Waves stay interior -> mass conserved despite transmissive edges.
    assert abs(h.sum() * dx - m0) / m0 < 1e-5

    wet = x[h > H_DRY]
    front_sim = float(wet.max()) if wet.size else -math.inf
    # Front sits within the tip-truncation band: not ahead of analytical, not far behind.
    assert front_sim <= front_ana + 2.0 * dx, f"front {front_sim:.1f} ahead of {front_ana:.1f}"
    assert front_sim >= 0.75 * front_ana, f"front {front_sim:.1f} lags {front_ana:.1f} badly"

    h_ref, _ = ritter(x, t_end, hL)
    fan = h_ref > 0.05 * hL  # score the clearly-wetted fan, skip the truncated tip
    nrmse = math.sqrt(np.mean((h[fan] - h_ref[fan]) ** 2)) / hL
    print(
        f"\n[hllc dry-bed Ritter] front_sim={front_sim:.1f} (ana {front_ana:.1f}) nRMSE={nrmse:.4f}"
    )
    assert nrmse < 0.05, f"dry-bed fan nRMSE {nrmse:.4f} too high"


def test_friction_damps_moving_slab_monotonically():
    """A moving slab on a flat bed decays to rest monotonically without reversing.

    Semi-implicit Manning divides momentum by ``D >= 1`` each step, so the interior
    velocity is strictly decreasing and never changes sign. The ``n=0`` control
    keeps its velocity: friction is the only thing removing momentum here.
    """
    ny, nx, dx = 4, 40, 10.0
    h0, u0 = 0.5, 2.0
    bed = np.zeros((ny, nx), dtype=np.float32)

    def run(n: float) -> list[float]:
        st = State.from_bed(bed, dx=dx, depth=h0, manning=n, device=DEV)
        hllc.arm_hllc(st)
        st.hu.assign(np.full((ny, nx), h0 * u0, dtype=np.float32))
        us = []
        for _ in range(25):
            dt = hllc.compute_dt(st, alpha=0.45, dt_max=5.0)
            hllc.step(st, dt=dt)
            u, _ = st.velocities_numpy()
            us.append(float(np.mean(u[1 : ny - 1, 10 : nx - 10])))
        return us

    us = run(0.05)
    print(f"\n[hllc friction] u: {u0:.3f} -> {us[0]:.3f} -> ... -> {us[-1]:.3f}")
    assert all(np.isfinite(us))
    # Strictly monotone decreasing, always positive (no sign reversal / overshoot).
    assert all(a > b for a, b in zip(us, us[1:], strict=False)), "velocity not monotone decreasing"
    assert us[-1] > 0.0 and us[0] < u0
    assert us[-1] < 0.5 * u0, "friction failed to bleed off momentum"

    us0 = run(0.0)
    assert abs(us0[-1] - u0) < 0.01 * u0, "frictionless slab should keep its velocity"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
