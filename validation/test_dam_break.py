"""Dam-break validation (M1 + M4 step 8, HANDOFF §10) on Warp's CPU backend.

Parametrized over **both** schemes behind the dispatch seam
(:func:`solver.core.schemes.get_scheme`) so one gate guards the coverage scheme
and the fidelity scheme (M4 plan §3 step 8):

* ``local_inertial`` -- the M1 Bates coverage scheme. It drops advective
  acceleration, so it smooths the shock and is a few % off on celerity; that is
  expected physics, held to a **loose** shape band.
* ``hllc_fv`` -- the M4 well-balanced HLLC fidelity scheme. It resolves the shock,
  so it is held to a much **tighter** band (nRMSE / front placement) that LI could
  not meet -- the discriminating check that HLLC actually beats LI on shocks.

Two gates with deliberately different tolerances (see M1 plan §5):

* **Mass conservation** -- hard, ``< 1e-6`` relative, for every case (both schemes,
  wet- and dry-bed). This is the real credibility gauge; continuity is conservative
  by construction and the float64/Kahan ledger proves it stays so. HLLC's
  conservative flux form holds the gate even through the ``wp.max(h, 0)``
  wetting-front clamp (dry-bed residual ~1e-8).
* **Wave shape vs analytical** -- scheme-dependent. The **wet-bed Stoker** case is
  the shape gate (small Manning friction damps the frontal oscillation). The
  **dry-bed Ritter** case -- the near-dry advancing tongue, each scheme's hardest
  regime -- runs its shape error as a reported, non-blocking diagnostic, but its
  mass gate still applies.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from solver.core.massbalance import MASS_GATE, MassLedger
from solver.core.schemes import get_scheme
from solver.core.state import State
from validation.analytical import ritter, stoker, stoker_front_position

wp.init()
DEV = "cpu"

# Per-scheme dam-break run + gate parameters. ``alpha`` is each scheme's CFL-like
# coefficient (Bates bound ~0.7 for LI; ``C*dx/(|u|+sqrt(g h))`` ~0.45 for HLLC).
# The wet-bed shape gates differ by design: LI smooths the shock (loose band, the
# established nRMSE ~0.074); HLLC resolves it and must clear a tight band LI cannot.
_SCHEMES = {
    "local_inertial": {"alpha": 0.7, "nrmse_max": 0.10, "front_max": 0.15},
    "hllc_fv": {"alpha": 0.45, "nrmse_max": 0.03, "front_max": 0.05},
}


def _cell_x(nx: int, dx: float) -> np.ndarray:
    """Cell-centre x-coordinates with the dam at x = 0 (between the two halves)."""
    return (np.arange(nx) + 0.5 - nx / 2.0) * dx


def _simulate_dambreak(
    h_l: float,
    h_r: float,
    *,
    scheme: str,
    alpha: float,
    nx: int,
    dx: float,
    t_end: float,
    manning_n: float,
) -> tuple[np.ndarray, np.ndarray, MassLedger]:
    """Run a 1-D dam break (single row) to ``t_end`` under ``scheme``; return (x, depth, ledger).

    The scheme is resolved through the same dispatch the run loop uses, so this
    exercises the real ``compute_dt``/``step`` seam -- LI and HLLC differ only in
    which module ``get_scheme`` returns (and the per-scheme ``alpha``).
    """
    sch = get_scheme(scheme)
    x = _cell_x(nx, dx)
    depth0 = np.where(x < 0.0, h_l, h_r).astype(np.float32)
    bed = np.zeros((1, nx), dtype=np.float32)
    st = State.from_bed(bed, dx=dx, depth=depth0.reshape(1, nx), manning=manning_n, device=DEV)
    ledger = MassLedger.from_state(st)

    t = 0.0
    while t < t_end - 1e-9:
        dt = sch.compute_dt(st, alpha=alpha, dt_max=t_end)
        dt = min(dt, t_end - t)
        sch.step(st, dt=dt)
        t += dt
    ledger.record(st, t)
    return x, st.h.numpy()[0].astype(np.float64), ledger


@pytest.mark.parametrize("scheme", list(_SCHEMES))
def test_wet_bed_stoker_is_the_gate(scheme: str):
    """Wet-bed dam break, both schemes: mass < 1e-6 AND wave shape within the band.

    LI clears the loose band (nRMSE ~0.074); HLLC must clear the tight band --
    beating LI on both nRMSE and front placement -- which it can only do by
    actually resolving the shock (M4 step 8 acceptance).
    """
    p = _SCHEMES[scheme]
    h_l, h_r = 1.0, 0.3
    nx, dx, t_end = 800, 0.5, 8.0
    x, h_sim, ledger = _simulate_dambreak(
        h_l, h_r, scheme=scheme, alpha=p["alpha"], nx=nx, dx=dx, t_end=t_end, manning_n=0.01
    )

    # --- Hard gate: mass conservation (both schemes) ---
    assert ledger.max_rel_error < MASS_GATE, f"mass leak {ledger.max_rel_error:.2e}"
    assert np.isfinite(h_sim).all()

    # --- Shape gate: wave shape vs Stoker (band is scheme-dependent) ---
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
        f"\n[{scheme} wet-bed] nRMSE={nrmse:.4f}  front_sim={front_sim:.1f} ref={front_ref:.1f} "
        f"err={front_err:.4f}  mass={ledger.max_rel_error:.2e}"
    )

    assert nrmse < p["nrmse_max"], f"{scheme} wave-shape nRMSE {nrmse:.4f} exceeds {p['nrmse_max']}"
    assert front_err < p["front_max"], (
        f"{scheme} front position error {front_err:.4f} exceeds {p['front_max']}"
    )


def _star_depth(h_l: float, h_r: float) -> float:
    from validation.analytical import _stoker_star_depth

    return _stoker_star_depth(h_l, h_r)


@pytest.mark.parametrize("scheme", list(_SCHEMES))
def test_dry_bed_ritter_diagnostic(scheme: str):
    """Dry-bed dam break, both schemes: mass gate enforced; wave shape reported.

    The near-dry advancing tongue is each scheme's hardest regime. Mass is a hard
    gate for both -- HLLC's conservative flux form holds it through the
    ``wp.max(h, 0)`` wetting-front clamp (residual ~1e-8) -- while the Ritter shape
    error is a non-blocking diagnostic with only a blow-up ceiling.
    """
    p = _SCHEMES[scheme]
    h_l = 1.0
    # Small non-zero downstream depth avoids the fully-dry front (LI's worst case)
    # while still exercising a near-dry advancing tongue; compared to Ritter (dry).
    h_r_seed = 1e-3
    nx, dx, t_end = 800, 0.5, 6.0
    x, h_sim, ledger = _simulate_dambreak(
        h_l, h_r_seed, scheme=scheme, alpha=p["alpha"], nx=nx, dx=dx, t_end=t_end, manning_n=0.02
    )

    # Hard gate still applies.
    assert ledger.max_rel_error < MASS_GATE, f"mass leak {ledger.max_rel_error:.2e}"
    assert np.isfinite(h_sim).all()

    # Diagnostic only: report shape error vs Ritter, do not fail on it.
    h_ref, _ = ritter(x, t_end, h_l)
    interior = np.abs(x) < 0.9 * (nx / 2.0 * dx)
    rmse = math.sqrt(np.mean((h_sim[interior] - h_ref[interior]) ** 2))
    nrmse = rmse / h_l
    print(
        f"\n[{scheme} dry-bed diagnostic] nRMSE={nrmse:.4f}  mass={ledger.max_rel_error:.2e} "
        f"(wave-shape non-blocking)"
    )
    # Only a sanity ceiling so a total blow-up still fails the diagnostic.
    assert nrmse < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
