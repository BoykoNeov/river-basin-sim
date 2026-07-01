"""Dam-break validation (M1, HANDOFF §10) on Warp's CPU backend.

Two gates with deliberately different tolerances (see M1 plan §5):

* **Mass conservation** -- hard, ``< 1e-6`` relative, for *both* the wet-bed and
  dry-bed runs. This is the real credibility gauge; continuity is conservative by
  construction and the float64/Kahan ledger proves it stays so.
* **Wave shape vs analytical** -- loose. Local-inertial drops advective
  acceleration, so it smooths the shock and is a few % off on celerity; that is
  expected physics. Enforced only for the **wet-bed Stoker** case (with small
  Manning friction to damp the frontal oscillation). The **dry-bed Ritter** case
  runs as a reported, non-blocking diagnostic -- its wetting front is LI's hardest
  regime -- but its mass gate still applies.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from solver.core.local_inertial import compute_dt, step
from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.state import State
from validation.analytical import ritter, stoker, stoker_front_position

wp.init()
DEV = "cpu"


def _cell_x(nx: int, dx: float) -> np.ndarray:
    """Cell-centre x-coordinates with the dam at x = 0 (between the two halves)."""
    return (np.arange(nx) + 0.5 - nx / 2.0) * dx


def _simulate_dambreak(
    h_l: float, h_r: float, *, nx: int, dx: float, t_end: float, manning_n: float
) -> tuple[np.ndarray, np.ndarray, MassLedger]:
    """Run a 1-D dam break (single row) to ``t_end``; return (x, depth, ledger)."""
    x = _cell_x(nx, dx)
    depth0 = np.where(x < 0.0, h_l, h_r).astype(np.float32)
    bed = np.zeros((1, nx), dtype=np.float32)
    st = State.from_bed(bed, dx=dx, depth=depth0.reshape(1, nx), device=DEV)
    ledger = MassLedger.from_state(st)

    t = 0.0
    while t < t_end - 1e-9:
        dt = compute_dt(st, alpha=0.7, dt_max=t_end)
        dt = min(dt, t_end - t)
        step(st, dt=dt, manning_n=manning_n)
        t += dt
    ledger.record(st, t)
    return x, st.h.numpy()[0].astype(np.float64), ledger


def test_wet_bed_stoker_is_the_gate():
    """Wet-bed dam break: mass < 1e-6 AND wave shape within the loose band."""
    h_l, h_r = 1.0, 0.3
    nx, dx, t_end = 800, 0.5, 8.0
    x, h_sim, ledger = _simulate_dambreak(h_l, h_r, nx=nx, dx=dx, t_end=t_end, manning_n=0.01)

    # --- Hard gate: mass conservation ---
    assert ledger.max_rel_error < MASS_GATE, f"mass leak {ledger.max_rel_error:.2e}"
    assert np.isfinite(h_sim).all()

    # --- Loose gate: wave shape vs Stoker ---
    h_ref, _ = stoker(x, t_end, h_l, h_r)
    # Compare only away from the boundaries (waves must not have reached them).
    interior = np.abs(x) < 0.9 * (nx / 2.0 * dx)
    rmse = math.sqrt(np.mean((h_sim[interior] - h_ref[interior]) ** 2))
    nrmse = rmse / (h_l - h_r)  # normalise by the dam-break amplitude

    # Front position: rightmost crossing of the mid-depth level between star & h_r.
    h_m_level = 0.5 * (h_r + _star_depth(h_l, h_r))
    front_sim = x[h_sim > h_m_level].max() if np.any(h_sim > h_m_level) else -np.inf
    front_ref = stoker_front_position(t_end, h_l, h_r)
    front_err = abs(front_sim - front_ref) / abs(front_ref)

    print(
        f"\n[wet-bed] nRMSE={nrmse:.3f}  front_sim={front_sim:.1f} ref={front_ref:.1f} "
        f"err={front_err:.3f}  mass={ledger.max_rel_error:.2e}"
    )

    assert nrmse < 0.10, f"wave-shape nRMSE {nrmse:.3f} too high"
    assert front_err < 0.15, f"front position error {front_err:.3f} too high"


def _star_depth(h_l: float, h_r: float) -> float:
    from validation.analytical import _stoker_star_depth

    return _stoker_star_depth(h_l, h_r)


def test_dry_bed_ritter_diagnostic():
    """Dry-bed dam break: mass gate enforced; wave shape reported, NOT enforced."""
    h_l = 1.0
    # Small non-zero downstream depth avoids the fully-dry front (LI's worst case)
    # while still exercising a near-dry advancing tongue; compared to Ritter (dry).
    h_r_seed = 1e-3
    nx, dx, t_end = 800, 0.5, 6.0
    x, h_sim, ledger = _simulate_dambreak(h_l, h_r_seed, nx=nx, dx=dx, t_end=t_end, manning_n=0.02)

    # Hard gate still applies.
    assert ledger.max_rel_error < MASS_GATE, f"mass leak {ledger.max_rel_error:.2e}"
    assert np.isfinite(h_sim).all()

    # Diagnostic only: report shape error vs Ritter, do not fail on it.
    h_ref, _ = ritter(x, t_end, h_l)
    interior = np.abs(x) < 0.9 * (nx / 2.0 * dx)
    rmse = math.sqrt(np.mean((h_sim[interior] - h_ref[interior]) ** 2))
    nrmse = rmse / h_l
    print(
        f"\n[dry-bed diagnostic] nRMSE={nrmse:.3f}  mass={ledger.max_rel_error:.2e} "
        f"(wave-shape non-blocking)"
    )
    # Only a sanity ceiling so a total blow-up still fails the diagnostic.
    assert nrmse < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
