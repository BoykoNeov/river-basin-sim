"""UK EA SC080035 Test 2 -- filling of floodplain depressions (M4 step 10).

The first of the M4 EA benchmark cases (SC080035, "Benchmarking of 2D Hydraulic
Modelling Packages", Appendix A.2). Test 2 evaluates a package's ability to
predict inundation extent and final flood depth in **low-momentum flow over a
complex topography with wetting and drying** -- precisely the regime the M4
step-10 conservative positivity limiter (:func:`solver.core.hllc._mass_beta`)
makes safe: the domain starts **dry**, water is injected at one corner, and the
scheme must fill the depressions and drive the rest dry while conserving mass to
the float64 gate. Before the limiter this case could not hold the mass gate (the
non-conservative ``wp.max(h, 0)`` clamp invented mass at every wetting front).

**Geometry (faithful to the report).** A 2000 m x 2000 m domain with a "flattened
egg-box": a 4 x 4 matrix of ~0.5 m deep depressions (product of N-S and W-E
sinusoids) on a mild slope -- 1:1500 N-S and 1:3000 W-E, ~2 m drop along the NW->SE
diagonal, so the NW corner is highest and the SE lowest. An inflow hydrograph
(peak 20 m3/s, ~85 min time base) enters on a 100 m line at the NW corner; all
edges are closed; the bed starts dry.

**Faithful-form, CI-tractable resolution.** The report specifies 20 m grid / 48 h.
This gate runs the *same* domain, slopes, egg-box wavelength and amplitude, inflow
and BCs at **40 m / 12 h** so it completes in a few seconds on Warp's CPU backend
(the pytest/CI target). The qualitative result is unchanged; a full 20 m / 48 h run
is a GPU demo (M4 step 11), not a CI gate.

**Pass criteria (qualitative + mass, per the report).** The report compares models
via figures, not tabulated envelopes, so -- per the plan (§3 step 10, §5: "don't
invent an nRMSE the report doesn't define") -- the gate asserts the report's
*qualitative* findings plus the always-on float64 mass gate:

* mass is conserved (``rel_error < MASS_GATE``); the domain is closed so the final
  stored volume equals the injected volume exactly (a banking-free identity);
* depths stay finite and non-negative through the wetting/drying (the limiter);
* the depressions fill (a clear majority hold water, the deepest to a good fraction
  of the 0.5 m relief) -- inundation actually develops;
* the **up-slope, top-right (NE) depressions stay dry** -- the report's headline
  finding that output points 15 & 16 remain dry in every model.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core import hllc
from solver.core.grid import H_DRY
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State
from solver.io.config import Inflow
from solver.processes.inflow import InflowInjector

wp.init()
DEV = "cpu"
_EDGES = ("north", "south", "east", "west")

DOMAIN = 2000.0  # m (square, per the report)
WAVELENGTH = 500.0  # m -> 4 depressions across each 2000 m side
AMP = 0.25  # m -> ~0.5 m peak-to-trough depression relief


def eggbox_dem(ny: int, nx: int, dx: float) -> np.ndarray:
    """SC080035 Test 2 "flattened egg-box" bed (row 0 = north, col 0 = west).

    Underlying slopes 1:1500 (N-S) and 1:3000 (W-E) -- NW highest, SE lowest, ~2 m
    diagonal drop -- with a 4 x 4 grid of wells (product of sinusoids, offset half a
    wavelength so 16 depressions sit inside the domain, the NE pair = points 15/16).
    """
    yy = np.arange(ny)[:, None] * dx  # metres south of the north edge
    xx = np.arange(nx)[None, :] * dx  # metres east of the west edge
    z_slope = (DOMAIN / 1500.0) * (1.0 - yy / DOMAIN) + (DOMAIN / 3000.0) * (1.0 - xx / DOMAIN)
    egg = (
        -AMP
        * np.cos(2 * np.pi * (xx - WAVELENGTH / 2) / WAVELENGTH)
        * np.cos(2 * np.pi * (yy - WAVELENGTH / 2) / WAVELENGTH)
    )
    return (z_slope + egg).astype(np.float32)


def depression_centres(dx: float) -> list[tuple[int, int]]:
    """(row, col) of the 16 depression centres at (250 + 500k) m, row-major N->S."""
    cs = [250.0 + WAVELENGTH * k for k in range(4)]
    return [(int(round(yc / dx)), int(round(xc / dx))) for yc in cs for xc in cs]


def test_ea_test2_floodplain_depressions_fill_conservatively():
    """EA Test 2: a dry basin fills its depressions, NE stays dry, mass gate holds."""
    dx = 40.0
    ny = nx = int(round(DOMAIN / dx))  # 50 x 50, faithful-form CI resolution
    bed = eggbox_dem(ny, nx, dx)
    st = State.from_bed(bed, dx=dx, depth=0.0, manning=0.03, device=DEV)
    st.set_open_boundaries({e: "closed" for e in _EDGES})  # all closed (report)

    # Inflow hydrograph on the 100 m line south of the NW corner (rows 0..nline-1,
    # col 0): peak 20 m3/s split evenly, triangular with ~85 min time base.
    nline = max(1, int(round(100.0 / dx)))
    peak, tb = 20.0, 85.0 * 60.0
    per = peak / nline
    inflows = [
        Inflow(cell=(i, 0), hydrograph=[(0.0, 0.0), (0.5 * tb, per), (tb, 0.0), (1.0e9, 0.0)])
        for i in range(nline)
    ]
    inj = InflowInjector(inflows, st.grid, DEV)
    ledger = MassLedger.from_state(st)

    # 12 h is past the transient for the assertions below (wet-count and the dry NE
    # corner are stable from ~12 h through 24 h); the field is still slowly creeping.
    # NB: the closed-domain mass residual grows with step count (float32 per-cell
    # update round-off, not a limiter defect: ~1e-7/3 h, ~3e-7/12 h, ~6e-7/24 h) --
    # re-check the gate margin before bumping this horizon or the resolution.
    t_end = 12.0 * 3600.0
    t = 0.0
    next_rec = 1800.0
    while t < t_end - 1e-9:
        dt = hllc.compute_dt(st, alpha=0.45, dt_max=30.0)
        dt = min(dt, t_end - t)
        ledger.add_inflow(inj.apply(st, t, dt))
        hllc.step(st, dt=dt)
        t += dt
        if t >= next_rec:  # sample the gate through the run, not only at the end
            ledger.record(st, t)
            next_rec += 1800.0
    rec = ledger.record(st, t)

    h = st.h.numpy()
    centres = depression_centres(dx)
    depths = np.array([h[i, j] for (i, j) in centres]).reshape(4, 4)  # rows N->S, cols W->E
    n_wet = int((depths > H_DRY).sum())
    print(
        f"\n[EA Test 2] steps->12h mass={ledger.max_rel_error:.2e} hmin={h.min():.2e}"
        f" hmax={h.max():.3f} wet_depressions={n_wet}/16 in={rec.inflow_cum:.0f}"
        f" vol={rec.volume:.0f}"
    )
    print("  depression depths (rows N->S, cols W->E):")
    for r in range(4):
        print("   " + "  ".join(f"{depths[r, c]:.3f}" for c in range(4)))

    assert np.isfinite(h).all(), "NaN/inf in depth -- wetting/drying instability"
    assert h.min() >= 0.0, f"depth went negative to {h.min():.3e}"
    # Mass gate: the load-bearing check. Closed domain => stored volume == injected
    # volume exactly, a banking-free identity the limiter must preserve as cells
    # wet and dry. (Pre-limiter, the clamp broke this.)
    assert ledger.max_rel_error < MASS_GATE, f"mass gate broke: {ledger.max_rel_error:.2e}"
    assert abs(rec.volume - rec.inflow_cum) < MASS_GATE * rec.inflow_cum

    # Inundation actually develops: a clear majority of depressions hold water and
    # the deepest (SE) well fills toward the ~0.5 m relief.
    assert n_wet >= 8, f"only {n_wet}/16 depressions filled; inundation did not develop"
    assert float(depths[3, 3]) > 0.2, f"deepest SE well only reached {depths[3, 3]:.3f} m"

    # The report's headline finding: the up-slope, top-right (NE) depressions stay
    # dry. Assert the NE-most pair (north row / east cols = points 15/16 analogue)
    # and the corner above them hold no water -- the discriminating extent check.
    assert float(depths[0, 3]) < H_DRY, "NE-corner depression (pt 16) should stay dry"
    assert float(depths[1, 3]) < H_DRY, "NE depression (pt 15) should stay dry"
    assert float(depths[0, 2]) < H_DRY, "top-right depression should stay dry"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
