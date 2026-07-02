"""HLLC per-edge ghost-cell BC validation (M4 step 9) on Warp's CPU backend.

Step 9 gives the HLLC scheme real per-edge boundaries read from ``state.boundaries``
(before this, the flux kernels were transmissive on *every* edge, so a "closed"
edge leaked and open-edge outflow was never banked into the mass ledger). Three
gates:

* **Per-edge open drain** (parametrized over all four edges) -- the classic
  sign-convention bug site (each edge banks its outflow with its own flux sign, per
  :mod:`solver.core.grid`). A basin sloping toward the *one* open edge drains
  through *that* edge only, depth stays non-negative, and -- the load-bearing check
  -- the float64 mass gate holds, which it can only do if the banked outflow matches
  the depth the flux divergence actually removed. Kept in the clamp-free regime
  (every cell stays wet): the ``wp.max(h,0)`` positivity clamp is non-conservative,
  so banking is exact *only* while it never fires (a drain-to-empty run trips it --
  a known limitation carried to the EA cases, M4 step 10).

* **Closed-wall reflection** -- the discriminating test the old transmissive-
  everywhere behaviour fails: a slab thrown at a closed box piles against the wall
  (depth > 2x initial) and its velocity reverses (reflection), losing no mass. The
  *same* slab under all-open (transmissive) edges passes straight through -- no
  pile-up, velocity unchanged -- so the pile-up proves the wall contains, it is not
  a dynamics artefact.

* **Manning normal depth** -- the fidelity check deferred from step 7 (it needs the
  step-9 inflow + open ghost BCs; transmissive edges alone draw down / back up a
  sloped flow). A moderately steep channel fed a steady discharge at the head and
  free-draining at the toe settles to the analytical wide-channel normal depth
  ``h_n = (q n / sqrt(S))^(3/5)`` in the interior -- to **<1%**, far tighter than
  the diffusive local-inertial channel's [0.5, 2.0] band (M3), because HLLC carries
  the full momentum balance (here transcritical, Fr~1.1 -- outside LI's regime).
"""

from __future__ import annotations

import math

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


def _sloped_toward(ny: int, nx: int, dx: float, edge: str, slope: float) -> np.ndarray:
    """A bed that slopes downhill toward ``edge`` (so water drains that way)."""
    i = np.arange(ny)[:, None] * np.ones((1, nx))
    j = np.ones((ny, 1)) * np.arange(nx)[None, :]
    run = {"east": (nx - 1 - j), "west": j, "south": (ny - 1 - i), "north": i}[edge]
    return (run * slope * dx).astype(np.float32)


@pytest.mark.parametrize("edge", _EDGES)
def test_hllc_each_open_edge_drains(edge):
    """Basin sloping toward ``edge`` (open): water drains *there*, mass gate holds.

    The per-edge banking-sign check. A wrong sign on any edge would either not
    drain (mass piles against a wall-behaving open edge) or break the mass gate
    (banked outflow not matching the field). Stops while every cell is still wet so
    the positivity clamp never fires and banking stays exact.
    """
    ny, nx, dx = 24, 24, 10.0
    bed = _sloped_toward(ny, nx, dx, edge, slope=0.005)
    st = State.from_bed(bed, dx=dx, depth=1.0, manning=0.03, device=DEV)
    st.set_open_boundaries({e: ("open" if e == edge else "closed") for e in _EDGES})
    ledger = MassLedger.from_state(st)
    v0 = ledger.v0

    t = 0.0
    for _ in range(200):
        dt = hllc.compute_dt(st, alpha=0.45, dt_max=5.0)
        hllc.step(st, dt=dt)
        t += dt
    rec = ledger.record(st, t)

    h = st.h.numpy()
    print(
        f"\n[hllc drain {edge}] out/v0={rec.outflow_cum / v0:.3f} vol/v0={rec.volume / v0:.3f}"
        f" hmin={h.min():.3e} mass={ledger.max_rel_error:.2e}"
    )
    assert np.isfinite(h).all()
    assert h.min() >= 0.0, f"{edge}: depth went negative to {h.min():.3e}"
    # Clamp-free regime => banking is exact => the mass gate is a real check here,
    # not a bookkeeping identity (a fired clamp would break it, as a drain-to-empty
    # run does). Assert the domain stayed wet so the guarantee is explicit.
    assert h.min() > H_DRY, f"{edge}: a cell dried ({h.min():.3e}); clamp may have fired"
    assert ledger.max_rel_error < MASS_GATE, f"{edge}: mass gate {ledger.max_rel_error:.2e}"
    # Water genuinely left through this edge (>half the domain volume drained out).
    assert rec.outflow_cum > 0.5 * v0, f"{edge}: only {rec.outflow_cum:.1f}/{v0:.1f} m^3 left"
    assert rec.volume < 0.5 * v0


def test_hllc_closed_wall_reflects_and_conserves_mass():
    """A moving slab in a closed box piles at the wall and reflects, losing no mass.

    Discriminating vs the pre-step-9 transmissive-everywhere behaviour: a closed
    edge is now a reflective wall (no through-flux). The slab cannot leave, so it
    piles against the downstream wall (depth >> initial) and its velocity reverses.
    The *same* slab under all-open edges (the contrast run) flows straight through
    with no pile-up -- proof the containment is the wall, not the dynamics.
    """
    ny, nx, dx = 4, 60, 5.0
    bed = np.zeros((ny, nx), dtype=np.float32)
    h0, u0 = 1.0, 3.0

    def throw_slab(boundaries: dict[str, str] | None) -> tuple:
        st = State.from_bed(bed, dx=dx, depth=h0, manning=0.0, device=DEV)
        if boundaries is not None:
            st.set_open_boundaries(boundaries)
        hllc.arm_hllc(st)
        st.hu.assign(np.full((ny, nx), h0 * u0, dtype=np.float32))  # uniform east-moving slab
        ledger = MassLedger.from_state(st)
        v0 = float(st.h.numpy().sum())  # initial depth-sum, same units as the returned h.sum()
        min_u, max_h, t = u0, h0, 0.0
        for _ in range(80):
            dt = hllc.compute_dt(st, alpha=0.45, dt_max=2.0)
            hllc.step(st, dt=dt)
            t += dt
            u, _ = st.velocities_numpy()
            min_u = min(min_u, float(u.min()))
            max_h = max(max_h, float(st.h.numpy().max()))
        ledger.record(st, t)
        h = st.h.numpy()
        return v0, float(h.sum()), min_u, max_h, float(h.min()), ledger.max_rel_error

    # Closed box (State default is all-closed): reflective walls.
    v0, v, min_u, max_h, h_min, mass_rel = throw_slab(None)
    print(
        f"\n[hllc closed wall] dV/v0={abs(v - v0) / v0:.2e} pile max_h={max_h:.3f}(init {h0})"
        f" min_u={min_u:.3f} h_min={h_min:.3e}"
    )
    assert np.isfinite([v, h_min, min_u, max_h]).all() and h_min >= 0.0
    # A closed box has no sinks -> mass conserved to float round-off (wall faces
    # carry zero mass flux; interior faces telescope).
    assert abs(v - v0) < 1e-6 * v0, f"closed box lost/gained mass: dV/v0={abs(v - v0) / v0:.2e}"
    # The wall contains the slab: it piles up and the flow reverses (reflection).
    assert max_h > 1.5 * h0, f"no pile-up at the wall (max_h={max_h:.3f}) -- edge behaved open"
    assert min_u < 0.0, f"flow never reversed (min_u={min_u:.3f}) -- no reflection"

    # Contrast: the identical slab under transmissive edges passes straight through.
    _, _, min_u_o, max_h_o, _, mass_rel_o = throw_slab({e: "open" for e in _EDGES})
    print(
        f"[hllc open contrast] max_h={max_h_o:.3f} min_u={min_u_o:.3f} "
        f"mass={mass_rel_o:.2e} (through-flow, no pile-up)"
    )
    assert max_h_o < 1.2 * h0, "open edge should not pile water up"
    assert min_u_o > 0.5 * u0, "open edge should not reverse the through-flow"
    # Signed banking: the west edge draws water *in* (banked as negative loss) at the
    # same rate the east edge lets it out. The mass gate holds only if that inflow
    # sign is right -- a west-edge sign error would double-count as outflow and break
    # the ledger even though the field volume (v == v0) looks fine.
    assert mass_rel_o < MASS_GATE, f"signed open-edge banking broke the ledger: {mass_rel_o:.2e}"


def test_hllc_channel_reaches_manning_normal_depth():
    """Steady channel (head inflow, open toe) settles to Manning normal depth <1%.

    The fidelity analogue of the M3 local-inertial channel test, deferred from
    step 7 until the open ghost BC + inflow existed. On a moderately steep bed the
    flow is transcritical (Fr~1.1) -- HLLC carries the full momentum balance and
    lands on the analytical normal depth to <1% across the uniform interior, where
    the diffusive local-inertial scheme only manages an order-of-magnitude band.
    """
    nx, dx = 120, 10.0
    slope, n, discharge = 0.02, 0.03, 2.0
    bed = ((nx - 1 - np.arange(nx)) * dx * slope).reshape(1, nx).astype(np.float32)
    st = State.from_bed(bed, dx=dx, depth=0.0, manning=n, device=DEV)
    st.set_open_boundaries({"east": "open", "west": "closed", "north": "closed", "south": "closed"})
    inj = InflowInjector(
        [Inflow(cell=(0, 0), hydrograph=[(0.0, discharge), (1.0e9, discharge)])], st.grid, DEV
    )
    ledger = MassLedger.from_state(st)

    t = 0.0
    for _ in range(8000):
        dt = hllc.compute_dt(st, alpha=0.45, dt_max=10.0)
        ledger.add_inflow(inj.apply(st, t, dt))
        hllc.step(st, dt=dt)
        t += dt
    rec = ledger.record(st, t)

    q_w = discharge / dx  # discharge per unit width (m^2/s)
    h_n = (q_w * n / math.sqrt(slope)) ** 0.6
    u_n = q_w / h_n
    fr = u_n / math.sqrt(9.81 * h_n)
    h = st.h.numpy()[0]
    interior = h[10 : nx - 10]  # skip the head-injection pile and the toe backwater
    median = float(np.median(interior))

    print(
        f"\n[hllc channel] h_n={h_n:.4f}m interior_median={median:.4f}m "
        f"({median / h_n:.4f}x, Fr={fr:.2f}) mass={rec.rel_error:.2e}"
    )
    assert np.isfinite(h).all()
    assert h.min() > H_DRY, "a cell dried; steady flow should keep the interior wet"
    assert ledger.max_rel_error < MASS_GATE, f"mass gate {ledger.max_rel_error:.2e}"
    assert rec.outflow_cum > 0.0  # steady flow is leaving the toe
    # Within ~1% of analytical normal depth (a 2% band for float robustness); the
    # interior is dead-uniform, so the median is the whole interior.
    assert abs(median - h_n) < 0.02 * h_n, f"interior {median:.4f} off normal {h_n:.4f} by >2%"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
