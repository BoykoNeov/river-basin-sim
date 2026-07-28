"""``fixed_stage`` boundary gates (M5) -- prescribed water surface on an HLLC edge.

The third ghost-cell type (after M4's reflective wall and transmissive open edge):
``eta_ghost = stage(t)``, zero-gradient velocity, and the Riemann solver decides
the direction. Four properties have to hold, and each can fail independently:

1. **At equilibrium it does nothing.** A domain whose surface already equals the
   stage must stay at rest -- if the ghost broke well-balancedness, every M4
   lake-at-rest guarantee would evaporate the moment a scenario used a stage edge.
2. **It lets water in**, driving the interior to the prescribed level, with the
   inflow banked so the float64 ledger balances (a stage edge is the first boundary
   in this project that can be a *source*; the banking is signed for exactly this).
3. **It lets water out**, including all the way to dry -- deliberately run into the
   positivity-limiter regime, since M4's carried limitation is that banking is exact
   only while the limiter does not rescale the banked face.
4. **It tracks a time-varying level**, which is what EA Test 1 needs.

Every case also carries the always-on float64 mass gate.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from solver.core import hllc
from solver.core.boundaries import stage_at
from solver.core.grid import H_DRY
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State

wp.init()
DEV = "cpu"
_EDGES = ("north", "south", "east", "west")


def _bc(**edges: str) -> dict[str, str]:
    """Per-edge map defaulting to closed."""
    return {e: edges.get(e, "closed") for e in _EDGES}


def _run(st: State, t_end: float, *, alpha: float = 0.45, dt_max: float = 5.0, record_every=None):
    """Step to ``t_end``, recording the mass ledger; returns the ledger."""
    ledger = MassLedger.from_state(st)
    every = record_every or max(t_end / 20.0, 1e-9)
    t, next_rec = 0.0, every
    while t < t_end - 1e-9:
        dt = min(hllc.compute_dt(st, alpha=alpha, dt_max=dt_max), t_end - t)
        hllc.step(st, dt=dt, t=t)
        t += dt
        if t >= next_rec - 1e-9:
            ledger.record(st, t)
            next_rec += every
    ledger.record(st, t)
    return ledger


def _ramp_bed(ny: int, nx: int, high: float = 1.0) -> np.ndarray:
    """Bed sloping from ``high`` at the west edge down to 0 at the east edge."""
    xx = np.linspace(high, 0.0, nx)[None, :].repeat(ny, axis=0)
    return xx.astype(np.float32)


def _ramp_toward(edge: str, ny: int, nx: int, high: float = 1.0) -> np.ndarray:
    """Bed sloping from ``high`` on the far side down to 0 at the named edge."""
    if edge == "east":
        prof = np.linspace(high, 0.0, nx)[None, :].repeat(ny, axis=0)
    elif edge == "west":
        prof = np.linspace(0.0, high, nx)[None, :].repeat(ny, axis=0)
    elif edge == "south":
        prof = np.linspace(high, 0.0, ny)[:, None].repeat(nx, axis=1)
    else:  # north
        prof = np.linspace(0.0, high, ny)[:, None].repeat(nx, axis=1)
    return prof.astype(np.float32)


# --- 1. equilibrium: a stage edge must not disturb a lake at rest ------------- #
def test_stage_edge_at_equilibrium_stays_at_rest():
    """Surface already at the prescribed stage => zero flux, by construction.

    The ghost copies the edge cell's depth (both sides see ``eta = stage``) and its
    velocity, so the Riemann problem is trivial and the flux is exactly 0. If this
    ever fails, the stage ghost has broken the well-balanced property and no
    lake-at-rest result on a stage-driven scenario can be trusted.
    """
    ny = nx = 24
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    bed = (0.02 * xx + 0.3 * np.sin(0.6 * xx) * np.cos(0.4 * yy)).astype(np.float32)
    surface = float(bed.max()) + 0.4
    h0 = (surface - bed.astype(np.float64)).astype(np.float32)

    st = State.from_bed(bed, dx=10.0, depth=h0, manning=0.0, device=DEV)
    st.set_open_boundaries(_bc(east="fixed_stage"), {"east": [(0.0, surface)]})
    ledger = _run(st, 600.0)

    u, v = st.velocities_numpy()
    worst = max(float(np.abs(u).max()), float(np.abs(v).max()))
    banked = st.loss_volume(st.grid.cell_area)
    print(f"\n[stage equilibrium] max|u,v|={worst:.2e} m/s  banked={banked:.2e} m3")
    assert worst < 1.0e-4, f"stage edge disturbed a lake at rest: {worst:.3e} m/s"
    assert ledger.max_rel_error < MASS_GATE
    # Nothing crossed the boundary either way (a wall would also pass the velocity
    # check; this is what says the *stage* ghost is inert at equilibrium).
    assert abs(banked) < 1e-3 * ledger.series[0].volume


# --- 2. inflow: the stage edge as a source ------------------------------------ #
@pytest.mark.parametrize("edge", _EDGES)
def test_stage_edge_fills_a_dry_domain_from_each_edge(edge):
    """Water enters from a stage above the bed and settles at the prescribed level.

    Run per-edge so a sign or index slip on any one of the four ghosts is caught
    (the same shape as the M4 per-edge open-drain gate).
    """
    ny = nx = 20
    bed = _ramp_toward(edge, ny, nx, high=1.0)  # the stage edge is the low end
    level = 0.5

    st = State.from_bed(bed, dx=10.0, depth=0.0, manning=0.03, device=DEV)
    st.set_open_boundaries(_bc(**{edge: "fixed_stage"}), {edge: [(0.0, level)]})
    ledger = _run(st, 1800.0, dt_max=2.0)

    h = st.h.numpy()
    eta = h + bed
    wet = h > 10 * H_DRY
    inflow_vol = -st.loss_volume(st.grid.cell_area)  # banked loss is negative here
    print(
        f"\n[stage fill {edge}] wet={wet.sum()}/{h.size} surface"
        f" mean={eta[wet].mean():.4f} max={eta[wet].max():.4f} in={inflow_vol:.1f} m3"
        f" mass={ledger.max_rel_error:.2e}"
    )
    assert np.isfinite(h).all() and h.min() >= 0.0
    assert ledger.max_rel_error < MASS_GATE, f"mass gate broke: {ledger.max_rel_error:.2e}"
    assert inflow_vol > 0.0, "no water entered through the stage edge"
    # Water fills to the prescribed level: the wetted area is where bed < level,
    # and its surface sits at the stage (a few mm of residual sloshing/friction).
    assert wet.sum() == pytest.approx((bed < level - 0.05).sum(), rel=0.25)
    assert float(np.abs(eta[wet] - level).max()) < 0.05, "interior did not reach the stage"


# --- 3. outflow, into the limiter regime -------------------------------------- #
def test_stage_edge_drains_and_the_banking_stays_exact():
    """A stage below the interior drains it -- including cells that go fully dry.

    Deliberately run into the positivity-limiter regime (M4's carried limitation is
    that open-boundary banking is exact only while the limiter does not rescale the
    banked face), and check the ledger with the *same* gate.
    """
    ny = nx = 20
    bed = _ramp_bed(ny, nx, high=1.0)
    h0 = np.full((ny, nx), 0.0, np.float32)
    h0[:] = np.maximum(0.9 - bed, 0.0)  # surface at 0.9 m over the whole ramp

    st = State.from_bed(bed, dx=10.0, depth=h0, manning=0.03, device=DEV)
    st.set_open_boundaries(_bc(east="fixed_stage"), {"east": [(0.0, 0.1)]})
    v0 = float(h0.astype(np.float64).sum()) * st.grid.cell_area
    ledger = _run(st, 3600.0, dt_max=2.0)

    h = st.h.numpy()
    eta = h + bed
    wet = h > 10 * H_DRY
    out = st.loss_volume(st.grid.cell_area)
    dry_frac = float((h <= H_DRY).mean())
    print(
        f"\n[stage drain] V0={v0:.1f} -> V={ledger.series[-1].volume:.1f} m3"
        f"  out={out:.1f}  dry={dry_frac:.0%}  mass={ledger.max_rel_error:.2e}"
    )
    assert np.isfinite(h).all() and h.min() >= 0.0
    assert ledger.max_rel_error < MASS_GATE, f"mass gate broke: {ledger.max_rel_error:.2e}"
    assert out > 0.5 * v0, "the stage edge barely drained anything"
    assert dry_frac > 0.4, "the drain never reached the dry/limiter regime it is here to test"
    # What is left sits at (or below) the prescribed stage.
    if wet.any():
        assert float(eta[wet].max()) < 0.1 + 0.05


# --- 4. a time-varying stage -------------------------------------------------- #
def test_interior_follows_a_time_varying_stage():
    """A rising-then-falling boundary level: the interior tracks it up and back.

    This is the EA Test 1 driving mechanism in miniature -- an open basin connected
    to the stage edge must follow the level, which is the control case the
    *disconnected* pond in that benchmark is contrasted against.
    """
    ny, nx = 12, 30
    bed = np.zeros((ny, nx), np.float32)  # flat: nothing can disconnect
    curve = [(0.0, 0.2), (900.0, 0.8), (1800.0, 0.2)]

    st = State.from_bed(bed, dx=10.0, depth=0.2, manning=0.03, device=DEV)
    st.set_open_boundaries(_bc(west="fixed_stage"), {"west": curve})

    samples: list[tuple[float, float, float]] = []
    ledger = MassLedger.from_state(st)
    t = 0.0
    while t < 1800.0 - 1e-9:
        dt = min(hllc.compute_dt(st, alpha=0.45, dt_max=2.0), 1800.0 - t)
        hllc.step(st, dt=dt, t=t)
        t += dt
        if len(samples) < int(t / 300.0):
            ledger.record(st, t)
            samples.append((t, stage_at(curve, t), float(st.h.numpy().mean())))
    ledger.record(st, t)

    print("\n[time-varying stage] t, prescribed, mean interior depth")
    for tt, s, hm in samples:
        print(f"   {tt:7.0f}  {s:.3f}  {hm:.3f}")
    assert ledger.max_rel_error < MASS_GATE

    peak = max(hm for _, _, hm in samples)
    final = samples[-1][2]
    assert peak > 0.45, f"interior never followed the stage up (peak mean depth {peak:.3f})"
    assert final < 0.5 * peak, f"interior never followed the stage back down ({final:.3f})"


# --- the stage curve itself ---------------------------------------------------- #
def test_stage_at_is_piecewise_linear_and_held_outside():
    """Held, not zeroed, outside the curve -- a water level does not vanish."""
    curve = [(0.0, 9.7), (3600.0, 10.35), (7200.0, 9.7)]
    assert stage_at(curve, -100.0) == 9.7
    assert stage_at(curve, 1800.0) == pytest.approx(10.025)
    assert stage_at(curve, 3600.0) == pytest.approx(10.35)
    assert stage_at(curve, 1e9) == 9.7  # held at the last value, not dropped to 0
    assert stage_at([(0.0, 4.2)], 555.0) == 4.2  # one-point curve = constant


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
